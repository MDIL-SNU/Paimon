"""RST parser for LAMMPS documentation using docutils."""

import sys
import json
import re
import warnings
from io import StringIO
from pathlib import Path

from docutils.core import publish_doctree
from docutils import nodes

from .models import LAMMPSDoc
from paimon.util.log import debug

warnings.filterwarnings("ignore", category=DeprecationWarning)


def preprocess_rst_content(content: str) -> str:
    """Preprocess RST content to handle Sphinx-specific roles.

    Convert Sphinx roles like :doc:`text <target>` to readable format.

    Args:
        content: Raw RST content

    Returns:
        Preprocessed RST content
    """
    # Pattern: :doc:`display text <target>` or :doc:`target`
    # Convert to: display text [→ target.rst] or target [→ target.rst]

    def replace_doc_role(match):
        full_text = match.group(1)

        # Check for display text format: `display <target>`
        inner_match = re.match(r"(.+?)\s*<(.+?)>", full_text)
        if inner_match:
            display_text = inner_match.group(1).strip()
            target = inner_match.group(2).strip()
            return f"{display_text} [→ {target}.rst]"
        else:
            # Simple format: `target`
            target = full_text.strip()
            return f"{target} [→ {target}.rst]"

    # Replace :doc:`...` patterns
    content = re.sub(r":doc:`([^`]+)`", replace_doc_role, content)

    # Also handle :ref:`...` similarly
    def replace_ref_role(match):
        full_text = match.group(1)
        inner_match = re.match(r"(.+?)\s*<(.+?)>", full_text)
        if inner_match:
            display_text = inner_match.group(1).strip()
            target = inner_match.group(2).strip()
            return f"{display_text} [→ {target}]"
        else:
            target = full_text.strip()
            return f"{target} [→ {target}]"

    content = re.sub(r":ref:`([^`]+)`", replace_ref_role, content)

    return content


class ParsingError:
    """Container for parsing errors."""

    def __init__(self, filepath: Path, section: str, error: str):
        self.filepath = filepath
        self.section = section
        self.error = error

    def __repr__(self):
        return (
            f"ParsingError({self.filepath.name}, {self.section}, {self.error[:50]})"
        )


class LAMMPSDocParser:
    """Parser for LAMMPS RST documentation files."""

    parsing_errors: list[ParsingError] = []

    @classmethod
    def dump_errors(cls, output_path: Path):
        """Dump all parsing errors to a JSON file.

        Args:
            output_path: Path to output JSON file
        """
        error_data = [
            {"file": str(err.filepath), "section": err.section, "error": err.error}
            for err in cls.parsing_errors
        ]

        with open(output_path, "w") as f:
            json.dump(error_data, f, indent=2)

        debug(f"Dumped {len(cls.parsing_errors)} parsing errors to {output_path}")

    @classmethod
    def clear_errors(cls):
        """Clear accumulated parsing errors."""
        cls.parsing_errors = []

    @staticmethod
    def parse_rst(filepath: Path) -> LAMMPSDoc:
        """Parse RST file and extract all sections.

        Args:
            filepath: Path to RST file

        Returns:
            Parsed LAMMPSDoc object
        """
        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read()

        # Preprocess to handle Sphinx roles
        content = preprocess_rst_content(content)

        filename = filepath.stem
        parts = filename.split("_", 1)
        command_type = parts[0] if len(parts) > 0 else "unknown"
        command_name = filename

        # The description parse works really bad
        # syntax = LAMMPSDocParser._extract_section(content, filepath, "Syntax")
        # examples = LAMMPSDocParser._extract_section(content, filepath, "Examples")
        # desc_paragraphs = LAMMPSDocParser._extract_description_paragraphs(
        #    content, filepath
        # )

        all_sections = LAMMPSDocParser._extract_all_sections(content, filepath)

        syntax = all_sections["Syntax"]
        examples = all_sections["Examples"]
        descriptions = all_sections["Description"]

        tmp = descriptions.split("\n\n")
        desc_paragraphs = []
        for para in tmp:
            if len(para.strip()) == 0:
                continue
            if para.startswith("```") and para.endswith("```"):
                desc_paragraphs.append(para)
            elif "[MATH]" in para:
                desc_paragraphs.append(para)
            else:
                desc_paragraphs.append(para.replace("\n", " "))

        return LAMMPSDoc(
            filepath=filepath,
            command_name=command_name,
            command_type=command_type,
            syntax=syntax,
            examples=examples,
            description_paragraphs=desc_paragraphs,
            full_content=content,
            all_sections=all_sections,
        )

    @staticmethod
    def _extract_all_sections(content: str, filepath: Path) -> dict[str, str]:
        """Extract ALL sections from document.

        Args:
            content: RST file content
            filepath: Path to RST file (for error reporting)

        Returns:
            Tuple of (dict mapping section names to content, error message if any)
        """
        try:
            old_stderr = sys.stderr
            sys.stderr = StringIO()

            settings_overrides = {
                "report_level": 5,
                "halt_level": 5,
                "file_insertion_enabled": True,
            }

            doctree = publish_doctree(content, settings_overrides=settings_overrides)
            sys.stderr = old_stderr

            all_sections = {}

            for section in doctree.findall(nodes.section):
                title = section.next_node(nodes.title)
                if title:
                    section_name = title.astext()
                    formatted_text = LAMMPSDocParser._format_node(
                        section, skip_title=True
                    )
                    all_sections[section_name] = formatted_text.strip()

            return all_sections

        except Exception as e:
            warnings.warn(f"{filepath}: Parse error: {str(e)}")
            return {}

    @staticmethod
    def _extract_section(content: str, filepath: Path, section_name: str) -> str:
        """Extract entire section content with proper formatting.

        Args:
            content: RST file content
            filepath: Path to RST file (for error reporting)
            section_name: Section name to extract

        Returns:
            Tuple of (section content as string, error message if any)
        """
        try:
            old_stderr = sys.stderr
            sys.stderr = StringIO()

            settings_overrides = {
                "report_level": 5,
                "halt_level": 5,
                "file_insertion_enabled": True,
            }

            doctree = publish_doctree(content, settings_overrides=settings_overrides)
            sys.stderr = old_stderr

            for section in doctree.findall(nodes.section):
                title = section.next_node(nodes.title)
                if title and section_name.lower() in title.astext().lower():
                    # Extract formatted content, skipping system_message nodes
                    formatted_text = LAMMPSDocParser._format_node(
                        section, skip_title=True
                    )
                    return formatted_text.strip()

            return f"No {section_name} section found"

        except Exception as e:
            warnings.warn(f"{filepath}: Parse error: {str(e)}")
            return f"Parse error in {section_name}"

    @staticmethod
    def _format_node(node, skip_title=False, indent=0) -> str:
        """Recursively format a node, preserving structure and converting RST roles.

        Args:
            node: docutils node to format
            skip_title: Skip title nodes
            indent: Current indentation level

        Returns:
            Formatted text string
        """
        # Skip system messages (errors)
        if isinstance(node, nodes.system_message):
            nodename = type(node).__name__
            tag = getattr(node, "tagname", None)
            preview = node.astext().strip().replace("\n", " ")[:80]
            debug(f"Error detacted: class={nodename} tag={tag}, preview={preview}")
            return ""

        # Skip title if requested
        if skip_title and isinstance(node, nodes.title):
            return ""

        # Handle different node types
        if isinstance(node, nodes.paragraph):
            # Process paragraph with inline elements
            text = LAMMPSDocParser._process_inline_nodes(node)
            return f"{text}\n\n"

        elif isinstance(node, nodes.math_block):
            math = node.astext().strip()
            return f"\n[MATH]\n{math}\n[/MATH]\n\n"

        elif isinstance(node, nodes.bullet_list):
            # Process bullet list
            result = []
            for item in node.findall(nodes.list_item):
                item_text = LAMMPSDocParser._format_node(item, indent=indent + 1)
                result.append(f"{'  ' * indent}• {item_text.strip()}")
            return "\n".join(result) + "\n\n"

        elif isinstance(node, nodes.enumerated_list):
            # Process numbered list
            result = []
            for i, item in enumerate(node.findall(nodes.list_item), 1):
                item_text = LAMMPSDocParser._format_node(item, indent=indent + 1)
                result.append(f"{'  ' * indent}{i}. {item_text.strip()}")
            return "\n".join(result) + "\n\n"

        elif isinstance(node, nodes.list_item):
            # Process list item children
            parts = []
            for child in node.children:
                if not isinstance(child, (nodes.bullet_list, nodes.enumerated_list)):
                    parts.append(LAMMPSDocParser._format_node(child, indent=indent))
            return "".join(parts).strip()

        elif isinstance(node, nodes.literal_block):
            # Code block
            return f"```\n{node.astext()}\n```\n\n"

        elif isinstance(node, nodes.note):
            # Note block
            content = ""
            for child in node.children:
                content += LAMMPSDocParser._format_node(child, indent=indent)
            return f"NOTE: {content.strip()}\n\n"

        elif isinstance(node, nodes.warning):
            # Warning block
            content = ""
            for child in node.children:
                content += LAMMPSDocParser._format_node(child, indent=indent)
            return f"WARNING: {content.strip()}\n\n"

        elif isinstance(node, nodes.transition):
            # --------------------- #
            return "\n\n"

        elif isinstance(node, nodes.target):
            return ""

        else:
            # Recursively process children
            if not node.children:
                nodename = type(node).__name__
                tag = getattr(node, "tagname", None)
                preview = node.astext().strip().replace("\n", " ")[:80]
                debug(f"Unhandled: class={nodename} tag={tag}, preview={preview}")

            result = []
            for child in node.children:
                result.append(
                    LAMMPSDocParser._format_node(
                        child, skip_title=skip_title, indent=indent
                    )
                )
            return "".join(result)

    @staticmethod
    def _extract_description_paragraphs(content: str, filepath: Path) -> list[str]:
        """Extract description section as list of paragraphs.

        Args:
            content: RST file content
            filepath: Path to RST file (for error reporting)

        Returns:
            Tuple of (list of paragraph strings, error message if any)
        """
        try:
            settings_overrides = {
                "report_level": 5,
                "halt_level": 5,
                "file_insertion_enabled": False,
            }

            doctree = publish_doctree(
                content,
                source_path=str(filepath),
                settings_overrides=settings_overrides,
            )

            for section in doctree.findall(nodes.section):
                title = section.next_node(nodes.title)
                if title and "description" in title.astext().lower():
                    paragraphs = []
                    for child in section.children:
                        if isinstance(child, nodes.paragraph):
                            text = child.astext().strip()
                            if text:
                                paragraphs.append(text)

                    paragraphs = paragraphs or ["No description paragraphs found"]
                    return paragraphs

            return ["No description section found"]

        except Exception as e:
            warnings.warn(f"{filepath}: Parse error: {str(e)}")
            return ["Parse error in Description"]

    @staticmethod
    def _process_inline_nodes(para_node) -> str:
        """Process inline nodes in a paragraph, converting RST roles to readable format.

        Args:
            para_node: paragraph node to process

        Returns:
            Processed text with readable link hints
        """
        result = []

        for child in para_node.children:
            if isinstance(child, nodes.Text):
                result.append(child.astext())

            elif isinstance(child, nodes.reference):
                # Internal reference/link
                refuri = child.get("refuri", "")
                text = child.astext()

                if refuri:
                    # External link
                    result.append(f"{text} ({refuri})")
                else:
                    # Internal reference - extract target from refid or refname
                    refid = child.get("refid", child.get("refname", ""))
                    if refid:
                        result.append(f"{text} [→ {refid}]")
                    else:
                        result.append(text)

            elif isinstance(child, nodes.title_reference):
                # :doc:`...` role
                text = child.astext()
                result.append(f"{text} [→ {text}.rst]")

            elif isinstance(child, nodes.emphasis):
                # *emphasis*
                result.append(f"*{child.astext()}*")

            elif isinstance(child, nodes.strong):
                # **strong**
                result.append(f"**{child.astext()}**")

            elif isinstance(child, nodes.literal):
                # ``literal``
                result.append(f"`{child.astext()}`")

            elif isinstance(child, nodes.math):
                # Math
                result.append(f"${child.astext()}$")

            else:
                # Fallback: just get text
                result.append(child.astext())

        return "".join(result)
