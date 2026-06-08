"""Documentation RAG systems.

Contains:
- LAMMPSRAGSystem: LAMMPS documentation with RST parsing and LLM selection
- ExpertRAGSystem: Simple RAG for expert knowledge text files

TODO: logger
"""

import asyncio
from pathlib import Path

import chromadb
from llama_index.core.schema import TextNode
from llama_index.core.llms import ChatMessage
from llama_index.embeddings.openai import OpenAIEmbedding

from .models import (
    DocEntry,
    LAMMPSDoc,
    RetrievalResult,
    SubQuery,
    SubQueryList,
    CommandSelection,
    ComplexRetrievalResult,
)
from .retrievers import DocRetrieverBuilder, LAMMPSRetrieverBuilder
from paimon.util.log import debug
from paimon import cfg
from paimon.llm import get_llm, run_agent_pipeline
from paimon.util.tool_factory import create_model_tool


RAG_QUERY_DECOMPOSER_PROMPT = """\
You are a query decomposition planner for a Retrieval-Augmented Generation (RAG) system.
The RAG system indexes *only* the LAMMPS documentation (commands, computes, fixes, etc.).
You NEVER answer the user’s scientific or simulation question directly.
Your only job is to transform an input task description into a small set of primitive search queries.

Goals:
- Take a high-level simulation or modeling task described in natural language.
- Identify the minimal set of *primitive, documentation-level* questions that, if answered, would give an agent all the information it needs to write correct LAMMPS input scripts.
- Each primitive query should be concrete enough that it can match specific LAMMPS commands, fixes, computes, or closely related reference pages.
- Focus primarily on the commands and building blocks that are essential to implement the requested behavior (e.g., fixes, computes, constraints, output/measurement), not on general theory or generic boilerplate.

Corpus:
- The RAG system only has access to LAMMPS documentation.
- This includes: command reference pages, compute/fix docs, how-to sections, and example scripts.
- It does NOT contain arbitrary research papers, textbooks, or code outside LAMMPS docs.

Trivial commands (do not target these):
- You are given a list of “trivial” commands that are *not* interesting for this decomposition task:
  {trivial_commands}
- Do NOT create subqueries whose primary purpose is to search for documentation on these trivial commands.

Constraints:
- Do NOT output any LAMMPS commands or code directly.
- Do NOT try to solve the original task.
- Do NOT describe how to run the simulation.
- Only describe what to search for in the documentation.
- SevenNet pair potential and its documents are already provided. Do not request it.

Subquery design guidelines:
- 1 to 4 subqueries is typical. Use more only if truly necessary.
- Each subquery must have:
  - A "role" chosen from: "dynamics", "constraints", "building_block", "record".
  - A "description" in one sentence explaining what kind of documentation we want.
  - A "query" string that will be sent *as-is* to the RAG retriever.

How to write a good query:
Describe what you want to do as if explaining to a colleague who will find the right command for you.

Examples
- "run simulation at constant pressure and temperature"  
- "calculate temperature of a subset of atoms"
- "remove atoms inside a spherical region"


Role semantics:
- "dynamics":
  Use this when the subquery is about how the system evolves in time or how the equations of motion are integrated.

- "constraints":
  Use this when the subquery is about restricting or constraining the motion or configuration of atoms.

- "building_block":
  Use this for structural and control commands that organize or construct (e.g., group, if)

- "record":
  Use this when the subquery is about measuring, averaging, or recording quantities during a run.

User task:
{task_description}

Now decompose this task into primitive search queries for the LAMMPS documentation, following all instructions above.
"""  # noqa: E501


RAG_LLM_SELECTOR_PROMPT = """\
You are a LAMMPS expert. Evaluate candidates and select the most relevant command.

Instructions:
1. Carefully read the query and each candidate's description paragraphs
2. Pay close attention to specific methods, algorithms, or approaches mentioned in the query
3. If the query specifies a particular method/algorithm/approach, ONLY select candidates that compatible with that specific method
4. If NONE of the candidates are appropriate for the query, set command_index to None
5. Otherwise, select the most relevant command and identify important paragraph numbers
6. If the query is not detailed, choose the simplest command among variants.

Return your selection using the CommandSelection tool.
User Query: "{query}"

Candidates:
{candidates_text}
"""  # noqa: E501


# Create tools for structured prediction
subquery_list_tool = create_model_tool(SubQueryList)
command_selection_tool = create_model_tool(CommandSelection)


class LAMMPSRAGSystem:
    """Optimized LAMMPS RAG system with ChromaDB metadata storage.

    Key optimization: All parsed data stored in ChromaDB metadata.
    No need to keep documents in memory or re-parse files.
    """

    def __init__(
        self,
        doc_dir: Path | str | None = None,
        embed_model: str | None = None,
        llm_model: str | None = None,
        use_hybrid: bool | None = None,
        chroma_path: str | None = None,
        force_rebuild: bool = False,
    ):
        """Initialize RAG system.

        Args:
            doc_dir: Directory containing LAMMPS RST files (uses cfg if None)
            embed_model: OpenAI embedding model name (uses cfg if None)
            llm_model: LLM model name for ranking (uses cfg if None)
            use_hybrid: Use hybrid search (uses cfg if None)
            chroma_path: Path to ChromaDB storage (uses cfg if None)
            force_rebuild: Force rebuild ChromaDB collection
        """
        self.doc_dir = Path(doc_dir or cfg.rag_config.lammps_docs_dir)
        self.use_hybrid = (
            use_hybrid if use_hybrid is not None else cfg.rag_config.use_hybrid
        )
        self.chroma_path = chroma_path or cfg.rag_config.chroma_db_path
        self.force_rebuild = force_rebuild

        embed_model_name = embed_model or cfg.rag_config.embed_model
        self.embed_model = OpenAIEmbedding(model=embed_model_name)

        self.llm_class = llm_model or cfg.rag_config.llm_model
        self.llm_kwargs = dict(
            metadata={"role": "LAMMPS rag"},
        )

        self.retriever_builder = None

    def build_index(self):
        """Build retrieval index.

        If ChromaDB exists and force_rebuild=False, loads existing index.
        Otherwise, parses RST files and builds new index.
        """
        debug(f"Initializing index (chroma_path: {self.chroma_path})...")

        self.retriever_builder = LAMMPSRetrieverBuilder(
            doc_dir=self.doc_dir,
            embed_model=self.embed_model,
            chroma_path=self.chroma_path,
        )

        # Get RST files if needed for building
        rst_files = None
        if self.force_rebuild or not self._check_collection_exists():
            rst_files = self._get_rst_files()
            debug(f"Found {len(rst_files)} RST files to index")

        self.retriever_builder.build_vector_index(
            rst_files=rst_files, force_rebuild=self.force_rebuild
        )

    def _check_collection_exists(self) -> bool:
        """Check if ChromaDB collection exists."""
        db = chromadb.PersistentClient(path=self.chroma_path)
        collections = [c.name for c in db.list_collections()]
        return "lammps_docs" in collections

    def _get_rst_files(self) -> list[Path]:
        """Get list of RST files to index."""
        all_rst_files = list(self.doc_dir.glob("*.rst"))
        return all_rst_files

    async def retrieve(
        self, query: str, top_k_stage1: int = 4, env=None, sub_wd=None
    ) -> tuple[RetrievalResult | None, list[LAMMPSDoc]]:
        """Retrieve relevant documents.

        TODO: change env & sub_wd to more general logger

        Args:
            query: User query
            top_k_stage1: Number of candidates to retrieve in stage 1

        Returns:
            Retrieval results (None if LLM rejected)
            List of query candidates (before LLM selection)
        """
        assert self.retriever_builder, "retriever_build is not initialized"
        if self.use_hybrid:
            retriever = self.retriever_builder.get_hybrid_retriever(
                top_k=top_k_stage1
            )
        else:
            retriever = self.retriever_builder.get_vector_retriever(
                top_k=top_k_stage1
            )

        nodes = retriever.retrieve(query)
        candidates = self.retriever_builder.get_documents_from_nodes(nodes)
        selected = await self._llm_select_and_quote(query, candidates)

        if env:
            retrieve_ev = dict(
                query=query,
                candidates=[
                    dict(
                        command=c.command_name,
                        key_paragraph=c.description_paragraphs[0],
                    )
                    for c in candidates
                ],
                selected=selected.command_name if selected else "LLM rejected",
            )
            env.append_json(
                key="retrieve_event",
                value=retrieve_ev,
                filename=".rag_events.json",
                sub_wd=sub_wd,
            )

        return selected, candidates

    async def _llm_select_and_quote(
        self,
        query: str,
        candidates: list[LAMMPSDoc],
        use_quote: bool = False,
    ) -> RetrievalResult | None:
        """LLM selects best command and quotes important paragraphs.

        Args:
            query: User query
            candidates: Candidate documents
            top_k: Number of results to return

        Returns:
            List of retrieval results
        """
        # Format candidates with numbered paragraphs
        candidates_text = ""
        for i, doc in enumerate(candidates, 1):
            candidates_text += f"[{i}] Command: {doc.command_name}\n\n"
            candidates_text += "Description paragraphs:\n"
            for j, para in enumerate(doc.description_paragraphs[:4], 1):
                candidates_text += f"  [{j}] {para}\n\n"
            if "Restrictions" in doc.all_sections:
                candidates_text += (
                    f"Restrictions:  {doc.all_sections['Restrictions']}\n\n"
                )
            candidates_text += "End of command descriptions.\n"

        prompt = RAG_LLM_SELECTOR_PROMPT.format(
            query=query, candidates_text=candidates_text
        )
        chat_history = [ChatMessage(role="user", content=prompt)]

        _, tool_calls, _ = await run_agent_pipeline(
            llm=get_llm(self.llm_class, **self.llm_kwargs),  # ensure stateless
            tools=[command_selection_tool],
            chat_history=chat_history,
            allow_parallel_tool_calls=False,
            tool_required=True,
            agent_name="rag_selector",
            metadata={"role": "rag_selector"},
        )
        if not tool_calls:
            raise ValueError("LLM selector no tool call error")

        selection = CommandSelection(**tool_calls[0].tool_kwargs)

        if selection.command_index is None:
            debug("LLM rejected all candidates (no appropriate match)")
            return None

        # Validate and use selection
        cmd_idx = selection.command_index - 1  # Convert to 0-based
        if not (0 <= cmd_idx < len(candidates)):
            debug(
                f"Warning: Invalid idx {selection.command_index} use first candidate"
            )
            cmd_idx = 0

        selected_doc = candidates[cmd_idx]

        if use_quote:
            para_nums = [
                p
                for p in selection.paragraph_numbers
                if 1 <= p <= len(selected_doc.description_paragraphs)
            ]
            if not para_nums:
                para_nums = [1]
            quoted_paras = []
            for num in sorted(para_nums):
                if 1 <= num <= len(selected_doc.description_paragraphs):
                    quoted_paras.append(selected_doc.description_paragraphs[num - 1])
            quoted_description = "\n\n".join(quoted_paras)
        else:
            quoted_description = "\n\n".join(selected_doc.description_paragraphs)

        return RetrievalResult(
            command_name=selected_doc.command_name,
            syntax=selected_doc.syntax,
            examples=selected_doc.examples,
            quoted_description=quoted_description,
            doc_raw=selected_doc,
        )

    def get_document_by_name(
        self, command_name: str, env=None, sub_wd=None
    ) -> RetrievalResult | None:
        """Directly lookup a document by exact command name.

        This is useful when you have a direct reference (e.g., from a link like
        `:doc:`dump custom<dump>`` → query "dump" directly).

        Args:
            command_name: Exact command name to lookup (e.g., "dump", "compute_msd")

        Returns:
            RetrievalResult if found, None otherwise
        """
        assert self.retriever_builder and self.retriever_builder.chroma_collection, (
            "retriever_build is not initialized"
        )
        # Try exact match
        results = self.retriever_builder.chroma_collection.get(
            where={"command_name": command_name},
            limit=1,
            include=["metadatas", "documents"],
        )

        ret = None
        ev = dict(query=command_name, selected=None)
        if results["metadatas"] and len(results["metadatas"]) > 0:
            metadata = results["metadatas"][0]
            doc_text = results["documents"][0]  # type: ignore

            node = TextNode(text=doc_text, id_="0", metadata=metadata)

            docs = self.retriever_builder.get_documents_from_nodes([node])
            if docs:
                doc = docs[0]
                ev["selected"] = doc.command_name
                ret = RetrievalResult(
                    command_name=doc.command_name,
                    syntax=doc.syntax,
                    examples=doc.examples,
                    quoted_description="\n\n".join(doc.description_paragraphs),
                    doc_raw=doc,
                )
        if env:
            env.append_json(
                key="get_document_by_name_event",
                value=ev,
                filename=".rag_events.json",
                sub_wd=sub_wd,
            )

        return ret

    async def retrieve_complex(
        self,
        task_description: str,
        trivial_commands: list[str] | None = None,
        top_k_per_subquery: int = 4,
    ) -> ComplexRetrievalResult:
        """Retrieve documentation for a complex, high-level task.

        This method implements a 3-stage retrieval process:
        - Stage 0: LLM decomposes task into primitive subqueries
        - Stage 1: Each subquery runs through standard retrieval
        - Stage 2: Results are formatted and combined

        Args:
            task_description: High-level task description
            trivial_commands: List of trivial commands to exclude from planning
            top_k_per_subquery: Number of candidates to retrieve per subquery (stage 1)

        Returns:
            ComplexRetrievalResult containing subqueries and formatted output
        """
        subqueries = await self._decompose_task(
            task_description,
            trivial_commands=trivial_commands,
        )

        tasks = []
        for subquery in subqueries:
            tasks.append(
                asyncio.create_task(
                    self.retrieve(
                        subquery.query,
                        top_k_stage1=top_k_per_subquery,
                    )
                )
            )

        all_results: list[
            tuple[RetrievalResult | None, list[LAMMPSDoc]]
        ] = await asyncio.gather(*tasks)
        all_results_selected = [sel for sel, _ in all_results]

        # Format combined output
        formatted_output = self._format_complex_results(
            subqueries=subqueries,
            results=all_results_selected,
            deduplicate=True,
        )

        return ComplexRetrievalResult(
            task_description=task_description,
            subqueries=subqueries,
            results=all_results_selected,
            formatted_output=formatted_output,
        )

    async def _decompose_task(
        self,
        task_description: str,
        trivial_commands: list[str] | None = None,
    ) -> list[SubQuery]:
        """Decompose a complex task into primitive subqueries using LLM.

        Args:
            task_description: High-level task description
            trivial_commands: List of trivial commands to exclude
            planner_prompt_path: Custom path to planner prompt file

        Returns:
            List of SubQuery objects
        """
        # Default trivial commands
        if trivial_commands is None:
            trivial_commands = [
                "units",
                "atom_style",
                "lattice",
                "create_box",
                "velocity",
                "thermo",
                "thermo_style",
                "dump",
                "run",
                "nvt, npt, nve",
                "minimize",
                "read_data",
                "timestep",
            ]
        trivial_commands_str = ", ".join(trivial_commands)

        prompt = RAG_QUERY_DECOMPOSER_PROMPT.format(
            trivial_commands=trivial_commands_str, task_description=task_description
        )
        chat_history = [ChatMessage(role="user", content=prompt)]

        # Use function calling for structured output
        _, tool_calls, _ = await run_agent_pipeline(
            llm=get_llm(self.llm_class, **self.llm_kwargs),  # ensure stateless
            tools=[subquery_list_tool],
            chat_history=chat_history,
            allow_parallel_tool_calls=False,
            tool_required=True,
            agent_name="rag_decomposer",
        )

        if not tool_calls:
            raise ValueError("No tool call received")

        result = SubQueryList(**tool_calls[0].tool_kwargs)
        return result.subqueries

    def _format_complex_results(
        self,
        subqueries: list[SubQuery],
        results: list[RetrievalResult | None],
        deduplicate: bool = True,
    ) -> str:
        """Format results from multiple subqueries into a single output.

        Args:
            subqueries: List of subqueries
            results: Dictionary mapping subquery index to retrieval results
            deduplicate: If True, remove duplicate command by its name

        Returns:
            Formatted string combining all results
        """
        output = []
        command_printed_set = set()

        for result in results:
            if not result:
                output.append("(No results found)")
                continue

            if result.command_name in command_printed_set and deduplicate:
                continue
            else:
                command_printed_set.add(result.command_name)

            output.append(f"<command: {result.command_name}>")
            output.append("<syntax>")
            output.append(result.syntax)
            output.append("</syntax>")
            output.append("<examples>")
            output.append(result.examples)
            output.append("</examples>")
            output.append("<quoted descriptions>")
            output.append(result.quoted_description)
            output.append("</quoted descriptions>")
            output.append(f"</command: {result.command_name}>")

        return "\n".join(output)


class ExpertRAGSystem:
    """Simple RAG system for expert knowledge files.

    Unlike LAMMPSRAGSystem, this:
    - Handles plain text files (no RST parsing)
    - Uses LLM-generated summaries as embedding keys
    - Returns top match directly (no LLM selection step)
    """

    def __init__(
        self,
        knowledge_dir: Path | str | None = None,
        embed_model: str | None = None,
        chroma_path: str | None = None,
        collection_name: str = "expert_knowledge",
        use_summary: bool = True,
        force_rebuild: bool = False,
    ):
        """Initialize expert RAG system.

        Args:
            knowledge_dir: Directory containing expert knowledge .txt files
            embed_model: OpenAI embedding model name
            chroma_path: Path to ChromaDB storage
            collection_name: ChromaDB collection name
            use_summary: Use LLM summary for embedding (else first paragraph)
            force_rebuild: Force rebuild ChromaDB collection
        """
        if knowledge_dir is None:
            # Default to paimon/knowledge/expert/
            knowledge_dir = (
                Path(__file__).parent.parent.parent / "knowledge" / "expert"
            )
        self.knowledge_dir = Path(knowledge_dir)
        self.chroma_path = chroma_path or cfg.rag_config.chroma_db_path
        self.collection_name = collection_name
        self.use_summary = use_summary
        self.force_rebuild = force_rebuild

        embed_model_name = embed_model or cfg.rag_config.embed_model
        self.embed_model = OpenAIEmbedding(model=embed_model_name)

        self.retriever_builder: DocRetrieverBuilder | None = None

    async def build_index(self) -> dict:
        """Build expert knowledge index.

        Returns:
            Build statistics dictionary
        """
        debug(f"Building expert knowledge index from: {self.knowledge_dir}")

        self.retriever_builder = DocRetrieverBuilder(
            embed_model=self.embed_model,
            chroma_path=self.chroma_path,
            collection_name=self.collection_name,
        )

        # Check if we need to rebuild
        if not self.force_rebuild and self._check_collection_exists():
            self.retriever_builder.build_vector_index(docs=None, force_rebuild=False)
            debug("Loaded existing expert knowledge index")
            return {"status": "loaded_existing"}

        # Find and process text files
        txt_files = list(self.knowledge_dir.glob("*.txt"))
        debug(f"Found {len(txt_files)} text files")

        docs = []
        for filepath in txt_files:
            if doc := (await self._process_file(filepath)):
                docs.append(doc)

        self.retriever_builder.build_vector_index(
            docs=docs, force_rebuild=self.force_rebuild
        )

        return {
            "status": "built",
            "total_files": len(txt_files),
            "indexed_docs": len(docs),
            "knowledge_dir": str(self.knowledge_dir),
        }

    def _check_collection_exists(self) -> bool:
        """Check if ChromaDB collection exists."""
        db = chromadb.PersistentClient(path=self.chroma_path)
        collections = [c.name for c in db.list_collections()]
        return self.collection_name in collections

    async def _process_file(self, filepath: Path) -> DocEntry | None:
        """Process a single text file into DocEntry."""
        try:
            content = filepath.read_text(encoding="utf-8")
            if not content.strip():
                debug(f"Skipping empty file: {filepath.name}")
                return None

            doc_id = filepath.stem
            title = self._extract_title(content, filepath)

            if filepath.name == "general.txt":
                # This is master dummy. Use pre-defined embedding key.
                embedding_key = """\
General atomistic simulation principles and best practices. This includes universal protocols for MD (Molecular Dynamics), DFT (Density Functional Theory), and MLIP (Machine Learning Interatomic Potentials). Use this when a request covers generic simulation setups, convergence criteria, unit conversions, or when a specific, specialized recipe for a particular material/system is not explicitly indexed.
"""  # noqa: E501
            # Generate embedding key
            elif self.use_summary:
                from paimon.rag.summarizer import summarize_for_embedding

                embedding_key = await summarize_for_embedding(
                    content=content, title=title
                )
            else:
                # Use first paragraph
                paragraphs = content.split("\n\n")
                first_para = paragraphs[0][:500] if paragraphs else ""
                embedding_key = f"Title: {title}\n\n{first_para}"

            return DocEntry(
                doc_id=doc_id,
                title=title,
                content=content,
                embedding_key=embedding_key,
                filepath=filepath,
                doc_type="expert",
            )
        except Exception as e:
            debug(f"Failed to process {filepath}: {e}")
            return None

    def _extract_title(self, content: str, filepath: Path) -> str:
        """Extract title from content or filename."""
        lines = content.strip().split("\n")
        # Check for explicit title line
        if lines and lines[0].startswith("Title:"):
            return lines[0].replace("Title:", "").strip()
        # Use first non-empty line if it looks like a title
        if lines and len(lines[0]) < 100 and not lines[0].startswith("#"):
            return lines[0].strip()
        # Fall back to filename
        return filepath.stem.replace("_", " ").title()

    async def retrieve(
        self,
        query: str,
        top_k: int = 3,
    ) -> DocEntry | None:
        """Retrieve relevant expert knowledge.

        Args:
            query: Search query
            top_k: Number of candidates to retrieve

        Returns:
            Best matching DocEntry, or None if no match
        """
        if not self.retriever_builder:
            raise ValueError("Index not built. Call build_index() first.")

        retriever = self.retriever_builder.get_vector_retriever(top_k=top_k)
        nodes = retriever.retrieve(query)

        if not nodes:
            return None

        docs = self.retriever_builder.get_documents_from_nodes(nodes)
        return docs[0] if docs else None

    async def retrieve_all(
        self,
        query: str,
        top_k: int = 3,
    ) -> list[DocEntry]:
        """Retrieve multiple expert knowledge documents.

        Args:
            query: Search query
            top_k: Number of results to return

        Returns:
            List of matching DocEntry objects
        """
        if not self.retriever_builder:
            raise ValueError("Index not built. Call build_index() first.")

        retriever = self.retriever_builder.get_vector_retriever(top_k=top_k)
        nodes = retriever.retrieve(query)

        return self.retriever_builder.get_documents_from_nodes(nodes)
