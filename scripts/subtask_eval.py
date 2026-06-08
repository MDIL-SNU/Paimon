import os
import sys
import asyncio
import json

from llama_index.core.callbacks import CallbackManager, TokenCountingHandler
from llama_index.core import Settings

import paimon.world as world
from paimon.models import Plan
from paimon.workflow.plan import generate_agent_prompt
from paimon.util.context import get_multi_agent_workflow_with_context


base_dir = os.path.dirname(__file__)
input_dir = os.path.join(base_dir, "inputfiles")
input_json = os.path.join(input_dir, "plan.json")


async def main():
    token_counter = TokenCountingHandler()
    Settings.callback_manager = CallbackManager([token_counter])
    print("Initial Token Count\n"
    "Embedding Tokens: ",
    token_counter.total_embedding_token_count,
    "\n",
    "LLM Prompt Tokens: ",
    token_counter.prompt_llm_token_count,
    "\n",
    "LLM Completion Tokens: ",
    token_counter.completion_llm_token_count,
    "\n",
    "Total LLM Token Count: ",
    token_counter.total_llm_token_count,
    "\n",
    )
    # initialization step
    env_id = world.new_environment()
    print("using folder : __%s, available as 'cd wd'" %(env_id))
    #print(input_json)
    with open(input_json, "r") as f:
        plan_dct = json.load(f)

    plan = Plan.model_validate(plan_dct)

    matches = [st for st in plan.subtasks if st.name == "Build MgO Supercell2"]
    if matches:
        task = matches[0]
    else:
        print("no match, use the first subtask as default")
        task = plan.subtasks[0] # tests the first subtask in default

    # check for input files to be needed
    missing = []
    for dep in task.dependencies:
        #print(dep)
        dep_task = next(st for st in plan.subtasks if st.name == dep)
        for out in dep_task.output:
            path = os.path.join(input_dir, out.filename)
            # print(out.filename)
            if not os.path.isfile(path):
                missing.append(out.filename)

    if missing:
        print("Missing dependency outputs:", missing)
        print("Evaluation could be wrong")

    else:
        print("All dependency outputs are present.")


    # subtasks and dependencies are checked
    print("agent name : %s" %(task.agent_name))
    print("subtask name : %s" %(task.name))

    #pirint("-----testing-----\n\n")

    # making links
    link = os.path.join(base_dir, "wd")

    if os.path.islink(link):
        os.unlink(link)
        print("Removed symlink wd")
    else:
        print("No symlink wd to remove")
    env = world.get_env(env_id)
    env.create_link_to_wd(base_dir)

    # copying files to the working directory
    for root, dirs, files in os.walk(input_dir):
        for fname in files:
            full_path = os.path.join(root, fname)
            with open(full_path, "r") as f:
                filecontent = f.read()
            # remote_path = relative path under inputfiles
            remote_path = os.path.relpath(full_path, input_dir)


            sub_wd = os.path.dirname(remote_path)

            if sub_wd:
                parts = sub_wd.split(os.sep)
                accum = ""
                for part in parts:
                    accum = os.path.join(accum, part) if accum else part
                    try:
                        #print(accum)
                        env.run(f"mkdir {accum} && chmod 775 {accum}", wrap_for_llm=True, sub_wd="")# noqa : E501
                    except Exception:
                        pass
            #print("writing step")
            try:
                env.write_file(
                    content    = filecontent,
                    remote_path= remote_path,
                    sub_wd     = "",
                )
                print(f"→ wrote {remote_path} into {link}")

            except Exception as e:
                print(f"Error writing {remote_path}: {e}", file=sys.stderr)
                continue
    print("copied all the files")
    print("-----testing-----\n\n")

    wf, ctx = get_multi_agent_workflow_with_context(
        task.agent_name, env_id=env_id, subtask=task, verbose=True
    )

    prompt = generate_agent_prompt(plan, task)
    resp = await wf.run(prompt, ctx=ctx)
    print(resp)
    print("Final Token Count\n"
    "Embedding Tokens: ",
    token_counter.total_embedding_token_count,
    "\n",
    "LLM Prompt Tokens: ",
    token_counter.prompt_llm_token_count,
    "\n",
    "LLM Completion Tokens: ",
    token_counter.completion_llm_token_count,
    "\n",
    "Total LLM Token Count: ",
    token_counter.total_llm_token_count,
    "\n",
    )

if __name__ == "__main__":
    asyncio.run(main())
