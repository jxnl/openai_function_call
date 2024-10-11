# Prompting Guide

Prompting requires understanding of techniques. Prompting enhances model performance.

We created examples of 58 prompting techniques<sup><a href="https://arxiv.org/abs/2406.06608">\*</a></sup> using `instructor`.

Prompting techniques are separated into the following categories:
- [Prompting Guide](#prompting-guide)
  - [Zero-Shot](#zero-shot)
  - [Few-Shot](#few-shot)
  - [Thought Generation](#thought-generation)
      - [Zero Shot](#zero-shot-1)
      - [Few Shot](#few-shot-1)
  - [Ensembling](#ensembling)
  - [Self-Criticism](#self-criticism)
  - [Decomposition](#decomposition)

Click links to learn about each method and how to apply them in prompts.

## Zero-Shot
How do we increase the performance of our model without any examples?

1. [Use Emotional Language](zero_shot/emotion_prompting.md)
2. [Assign a Role](zero_shot/role_prompting.md)
3. [Define a Style](zero_shot/style_prompting.md)
4. [Auto-Refine The Prompt](zero_shot/s2a.md)
5. [Simulate a Perspective](zero_shot/simtom.md)
6. [Clarify Ambiguous Information](zero_shot/rar.md)
7. [Ask Model To Repeat Query](zero_shot/re2.md)
8. [Generate Follow-Up Questions](zero_shot/self_ask.md)

## Few-Shot

How do we choose effective examples to include in our prompt?

1. [Generate Examples](few_shot/example_generation/sg_icl.md)
2. [Order Examples](few_shot/example_ordering.md)
3. [Choose Examples Similar to the Query (KNN)](few_shot/exemplar_selection/knn.md)
4. [Choose Examples Similar to the Query (Vote-K)](few_shot/exemplar_selection/vote_k.md)

## Thought Generation

How do we encourage our model to mimic human-like reasoning?

#### Zero Shot

1. [Auto-Generate Chain-Of-Thought Examples](thought_generation/chain_of_thought_zero_shot/analogical_prompting.md)
2. [First Ask a Higher-Level Question](thought_generation/chain_of_thought_zero_shot/step_back_prompting.md)
3. [Encourage Analysis](thought_generation/chain_of_thought_zero_shot/thread_of_thought.md)
4. [Encourage Structural Reasoning](thought_generation/chain_of_thought_zero_shot/tab_cot.md)

#### Few Shot
5. [Annotate Only Uncertain Examples](thought_generation/chain_of_thought_few_shot/active_prompt.md)
6. [Choose Diverse Examples](thought_generation/chain_of_thought_few_shot/auto_cot.md)
7. [Choose Complex Examples](thought_generation/chain_of_thought_few_shot/complexity_based.md)
8. [Include Incorrect Demonstrations](thought_generation/chain_of_thought_few_shot/contrastive.md)
9. [Choose Similar, Auto-Generated, High-Certainty Chain-Of-Thought Reasonings](thought_generation/chain_of_thought_few_shot/memory_of_thought.md)
10. [Choose the Most Certain Reasoning](thought_generation/chain_of_thought_few_shot/uncertainty_routed_cot.md)
11. [Generate Template-Based Prompts](thought_generation/chain_of_thought_few_shot/prompt_mining.md)

## Ensembling

How can we use multiple prompts and aggregate their responses?

1. [Build a Set of Consistent, Diverse Examples](ensembling/cosp.md)
2. [Batch In-Context Examples](ensembling/dense.md)
3. [Verify Individual Reasoning Steps](ensembling/diverse.md)
4. [Maximize Information Between Input and Output](ensembling/max_mutual_information.md)
5. [Merge Multiple Chains-Of-Thought](ensembling/meta_cot.md)
6. [Use Specialized Experts](ensembling/more.md)
7. [Choose The Most Consistent Reasoning](ensembling/self_consistency.md)
8. [Choose The Most Consistent Reasioning (Universal)](ensembling/universal_self_consistency.md)
9. [Use Task-Specific Example Selection](ensembling/usp.md)
10. [Paraphrase The Prompt](ensembling/prompt_paraphrasing.md)

## Self-Criticism

How can a model verify or critique its own response?

1. [Generate Verification Questions](self_criticism/chain_of_verification.md)
2. [Ask If the Answer is Correct](self_criticism/self_calibration.md)
3. [Generate Feedback and Auto-Improve](self_criticism/self_refine.md)
4. [Score Multiple Candidate Solutions](self_criticism/self_verification.md)
5. [Reconstruct The Problem](self_criticism/reversecot.md)
6. [Generate Possible Steps](self_criticism/cumulative_reason.md)

## Decomposition

How can we break down complex problems? How do we solve subproblems?

1. [Implement Subproblems As Functions](decomposition/decomp.md)
2. [Use Natural and Symbolic Language](decomposition/faithful_cot.md)
3. [Solve Increasingly Complex Subproblems](decomposition/least_to_most.md)
4. [Generate a Plan](decomposition/plan_and_solve.md)
5. [Use Code As Reasoning](decomposition/program_of_thought.md)
6. [Recursively Solve Subproblems](decomposition/recurs_of_thought.md)
7. [Generate a Skeleton](decomposition/skeleton_of_thought.md)
8. [Search Through Subproblems](decomposition/tree-of-thought.md)

<sup id="ref-asterisk">\*</sup>: [The Prompt Report: A Systematic Survey of Prompting Techniques](https://arxiv.org/abs/2406.06608)
