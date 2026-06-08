#!/usr/bin/python3
import os.path as osp
import json
import matplotlib.pyplot as plt
import numpy as np
import argparse
import sys


def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(
        description="Paimon Trajectory Visualization"
    )
    parser.add_argument("tok_json", help="Path to the token usage JSON file")
    parser.add_argument(
        "--action_space_json",
        default=f"{osp.join(osp.dirname(__file__), 'action_space.json')}",
        help="Path to the action space JSON file",
    )
    parser.add_argument(
        "--output_file",
        default="./agent_traj.png",
        help="Output file name for the figure (e.g., trajectory.png)",
    )

    args = parser.parse_args()

    try:
        # Load the data
        with open(args.tok_json, "r") as f:
            tok_data = json.load(f)

        with open(args.action_space_json, "r") as f:
            action_space_data = json.load(f)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"Error parsing JSON file: {e}")
        sys.exit(1)

    # Extract trajectory data
    items = tok_data["items"]
    steps = list(range(1, len(items) + 1))

    # Get the agent name (all entries are 'plan' in this case)
    agent_name = items[0]["name"]

    if agent_name not in action_space_data:
        print(f"Error: Agent '{agent_name}' not found in action space data")
        sys.exit(1)

    available_actions = action_space_data[agent_name]

    # Create action space mapping
    action_to_index = {action: i for i, action in enumerate(available_actions)}

    # Extract data for plotting
    tool_calls = [item["tool_call"][0] for item in items]

    # Check if all tool calls are valid
    invalid_calls = [call for call in tool_calls if call not in action_to_index]
    if invalid_calls:
        print(f"Warning: Invalid tool calls found: {set(invalid_calls)}")
        print(f"Available actions: {available_actions}")

    action_indices = [action_to_index.get(tool_call, 0) for tool_call in tool_calls]
    reasoning_tokens = [item["reasoning_tokens"] for item in items]
    output_tokens = [item["output_tokens"] for item in items]
    non_reasoning_tokens = [
        out - reason for out, reason in zip(output_tokens, reasoning_tokens)
    ]
    total_costs = [item["total_cost"] for item in items]

    # Create the figure with two subplots (1/4 size)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6.3, 4.5), sharex=True)

    # Upper panel - Action Space (dotted line + markers)
    colors = ["#FF6B6B", "#4ECDC4", "#45B7D1", "#96CEB4", "#FFEAA7", "#DDA0DD"]
    color_map = {
        action: colors[i % len(colors)] for i, action in enumerate(available_actions)
    }
    marker_colors = [color_map.get(tool_call, "#808080") for tool_call in tool_calls]

    # Plot line with markers
    ax1.plot(
        steps, action_indices, linestyle=":", linewidth=1.5, color="gray", alpha=0.7
    )

    # Add markers with different colors for each action
    for i, (step, action_idx, color) in enumerate(
        zip(steps, action_indices, marker_colors)
    ):
        ax1.scatter(
            step,
            action_idx,
            c=color,
            s=50,
            alpha=0.8,
            edgecolor="black",
            linewidth=0.5,
            zorder=3,
        )

    ax1.set_title(
        f"{agent_name.capitalize()} Trajectory",
        fontsize=10,
        fontweight="bold",
        pad=10,
    )
    ax1.set_yticks(range(len(available_actions)))
    ax1.set_yticklabels(available_actions, fontsize=8)
    ax1.grid(True, alpha=0.3, axis="y")

    # Add padding to y-axis limits for better marker positioning
    ax1.set_ylim(-0.5, len(available_actions) - 0.5)

    # Add action labels next to markers (simplified for small size)
    for i, (step, action_idx, tool_call) in enumerate(
        zip(steps, action_indices, tool_calls)
    ):
        if i % 2 == 0:  # Only label every other point to avoid crowding
            ax1.text(
                step,
                action_idx + 0.2,
                tool_call,
                ha="center",
                va="bottom",
                rotation=45,
                fontsize=6,
                alpha=0.7,
            )

    # Lower panel - Token count and cost (adjusted for smaller size)
    width = 0.35
    x_pos = np.array(steps)

    # Left y-axis for tokens
    bars2 = ax2.bar(
        x_pos - width / 2,
        reasoning_tokens,
        width,
        label="Reasoning Tokens",
        color="#E74C3C",
        alpha=0.8,
    )
    bars3 = ax2.bar(
        x_pos - width / 2,
        non_reasoning_tokens,
        width,
        bottom=reasoning_tokens,
        label="Non-reasoning Output Tokens",
        color="#3498DB",
        alpha=0.8,
    )

    ax2.set_xlabel("Agent Step", fontsize=8, fontweight="bold")
    ax2.set_ylabel("Token Count", fontsize=8, fontweight="bold", color="blue")
    ax2.tick_params(axis="y", labelcolor="blue", labelsize=8)
    ax2.tick_params(axis="x", labelsize=8)

    # Right y-axis for cost
    ax2_right = ax2.twinx()
    bars4 = ax2_right.bar(
        x_pos + width / 2,
        total_costs,
        width,
        label="Total Cost ($)",
        color="#2ECC71",
        alpha=0.8,
    )
    ax2_right.set_ylabel("Cost ($)", fontsize=8, fontweight="bold", color="green")
    ax2_right.tick_params(axis="y", labelcolor="green", labelsize=8)

    # Simplified value labels for smaller size (only show every other bar)
    for i, (bar2, bar3, reason, non_reason) in enumerate(
        zip(bars2, bars3, reasoning_tokens, non_reasoning_tokens)
    ):
        if i % 3 == 0:  # Only label every 3rd bar to avoid crowding
            # Reasoning tokens label
            if reason > 0:
                ax2.text(
                    bar2.get_x() + bar2.get_width() / 2.0,
                    reason / 2,
                    f"{reason}",
                    ha="center",
                    va="center",
                    fontsize=4,
                    fontweight="bold",
                )

    # Simplified cost labels
    for i, (bar, cost) in enumerate(zip(bars4, total_costs)):
        if i % 3 == 0:  # Only label every 3rd bar
            height = bar.get_height()
            ax2_right.text(
                bar.get_x() + bar.get_width() / 2.0,
                height / 2,
                f"${cost:.3f}",
                ha="center",
                va="center",
                fontsize=4,
                fontweight="bold",
                rotation=90,
            )

    # Set x-axis
    ax2.set_xlim(0.5, len(steps) + 0.5)
    ax2.set_xticks(steps)
    ax2.grid(True, alpha=0.3, axis="y")

    # Create legends (adjusted for smaller size)
    legend_elements_tokens = [bars2, bars3]
    legend_elements_cost = [bars4]

    ax2.legend(
        legend_elements_tokens,
        ["Reasoning Tokens", "Non-reasoning Output Tokens"],
        loc="upper left",
        bbox_to_anchor=(0, 1),
        fontsize=8,
    )
    ax2_right.legend(
        legend_elements_cost,
        ["Total Cost ($)"],
        loc="upper right",
        bbox_to_anchor=(1, 1),
        fontsize=8,
    )

    # Adjust layout
    plt.tight_layout()

    # Simplified summary statistics for smaller figure
    total_tokens = sum(output_tokens)
    total_cost = sum(total_costs)
    total_reasoning = sum(reasoning_tokens)

    summary_text = f"""Steps: {len(steps)} | Agent: {agent_name}
Tokens: {total_tokens:,} | Cost: ${total_cost:.3f}
Reasoning: {total_reasoning / total_tokens * 100:.0f}%"""

    plt.figtext(
        0.02,
        0.02,
        summary_text,
        fontsize=5,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.8),
    )

    # Save the figure
    try:
        plt.savefig(args.output_file, dpi=300, bbox_inches="tight")
        print(f"Figure saved as: {args.output_file}")
    except Exception as e:
        print(f"Error saving figure: {e}")
        sys.exit(1)

    # Print summary statistics
    print("Visualization completed!")
    print(f"Total steps: {len(steps)}")
    print(f"Agent: {agent_name}")
    print(f"Available actions: {available_actions}")
    print(f"Actions used: {set(tool_calls)}")
    print(f"Total cost: ${total_cost:.4f}")
    print(f"Total output tokens: {total_tokens:,}")


if __name__ == "__main__":
    main()
