import paimon.agent.agent_registry as regi

regi.list_agents()

dct = {rr: [] for rr in regi.list_agents()}

for name in regi.list_agents():
    rr = regi.get_agent_config(name)
    dct[name] = [t.metadata.name for t in rr.tools]
    dct[name].append("abort_task")
    dct[name].append("complete_task")

dct["plan"] = ["outline_plan", "create_subtask", "ask_agent", "discard_subtasks_since", "abort_task", "complete_task"]



