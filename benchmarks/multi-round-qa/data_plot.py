import json
import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

with open('ShareGPT_stats.json', 'r', encoding='utf-8') as file:
    data = json.load(file)
    all_num_rounds = data['num_rounds']
    all_prompt_lengths = data['prompt_lengths']
    all_output_lengths = data['output_lengths']
    all_prompt_lengths_with_history = data['prompt_lengths_with_history']
    plot_data_for_df = data['plot_data_for_df']

sns.set_theme()
os.makedirs('results', exist_ok=True)

if all_num_rounds:
    plt.figure(figsize=(10, 6))
    sns.ecdfplot(x=all_num_rounds)
    plt.title('CDF of Number of Turns')
    plt.xlabel('Number of Turns')
    plt.ylabel('Proportion')
    plt.grid(True)
    plt.savefig('results/cdf_num_turns.png')
    plt.close()
    print("Saved cdf_num_turns.png")

if all_prompt_lengths or all_output_lengths or all_prompt_lengths_with_history:
    plot_data_long = []
    if all_prompt_lengths:
        plot_data_long.extend([{
            'length': l,
            'type': 'Prompt'
        } for l in all_prompt_lengths])
    if all_output_lengths:
        plot_data_long.extend([{
            'length': l,
            'type': 'Output'
        } for l in all_output_lengths])
    if all_prompt_lengths_with_history:
        plot_data_long.extend([{
            'length': l,
            'type': 'Prompt with History'
        } for l in all_prompt_lengths_with_history])

    if plot_data_long:
        df_lengths = pd.DataFrame(plot_data_long)
        plt.figure(figsize=(10, 6))
        sns.ecdfplot(data=df_lengths, x='length', hue='type')
        plt.xscale('log')
        plt.title('CDF of Token Lengths')
        plt.xlabel('Token Length (log scale)')
        plt.ylabel('Proportion')
        plt.grid(True)
        plt.savefig('results/cdf_lengths.png')
        plt.close()
        print("Saved cdf_lengths.png")

if plot_data_for_df:
    df = pd.DataFrame(plot_data_for_df)
    df['total_length'] = df['total_prompt_length'] + df['total_output_length']
    turns_to_plot = [2, 6, 14]

    available_turns = sorted(df['num_round'].unique())
    turns_to_plot = [t for t in turns_to_plot if t in available_turns]

    if turns_to_plot:
        filtered_df = df[df['num_round'].isin(turns_to_plot)]

        if not filtered_df.empty:
            plt.figure(figsize=(10, 6))
            sns.boxplot(
                data=filtered_df,
                x='num_round',
                y='total_length',
                showfliers=False)
            plt.title('Distribution of Total Prompt Lengths by Number of Turns')
            plt.xlabel('Number of Turns')
            plt.ylabel('Total Prompt Length (tokens)')
            plt.grid(True)
            plt.tight_layout()
            plt.savefig('results/prompt_length_dist_by_turns.png')
            plt.close()
            print("Saved prompt_length_dist_by_turns.png")

    # New plot for cumulative bar plot
    required_cols = [
        'num_round', 'last_history_length', 'last_prompt_length',
        'last_output_length'
    ]
    if all(col in df.columns for col in required_cols):
        plot_data_agg = df[df['num_round'].isin(turns_to_plot)]
        plot_data_agg = plot_data_agg.groupby('num_round')[[
            'last_history_length', 'last_prompt_length', 'last_output_length'
        ]].mean()

        # Normalize the data to sum to 1 for each round
        plot_data_normalized = plot_data_agg.div(plot_data_agg.sum(axis=1),
                                                  axis=0)
        plot_data_normalized.columns = ['History', 'Prompt', 'Output']

        if not plot_data_normalized.empty:
            plot_data_normalized.plot(
                kind='bar',
                stacked=True,
                figsize=(10, 6),
                color=['skyblue', 'salmon', 'lightgreen'])
            plt.title('Average Composition of Last Turn by Number of Turns')
            plt.xlabel('Number of Turns')
            plt.ylabel('Proportion')
            plt.xticks(rotation=0)
            plt.grid(axis='y', linestyle='--')
            plt.legend(title='Length Type')
            plt.tight_layout()
            plt.savefig('results/last_turn_composition.png')
            plt.close()
            print("Saved last_turn_composition.png")
