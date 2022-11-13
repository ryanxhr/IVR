
## Revision Summary
We thank all the reviewers for the detailed and constructive comments. We have revised the paper to address the concerns of the reviewers. The summary of changes in the updated version of the paper is as follows:

- We re-run IQL and CQL on all datasets and report the best scores in Table 1.
- We add the standard deviation of IQL and CQL in Table 1.
- We list the per-dataset $\alpha$ of SQL in Table 7.
- We provide the sensitivity of $\alpha$ in SQL in Table 5 and Table 6, and provide the sensitivity of $\tau$ in IQL in Table 8.
- We use rliable library to validate the claim for SOTA results of SQL in Figure 7.
- We list the runtime of all algorithms in Table 9.
- We derive another practical new offline RL algorithm, EQL, by applying the reverse KL divergence in the IVR framework in Appendix A, we find EQL is an "implicit" version of AWR/AWAC while has a strong empirical performance.
- We add a whole section to elaborate how sparsity benefits value learning in SQL, in both tabular setting and continuous action setting, in Appendix B.
- We add a "Reproducibilty Statement" section to ensure reproducibilty.

We also want to highlight that the contributions of our paper is not only proposing a new effective offline RL algorithm, but more importantly, we propose a general Implicit Value Regularization framework, which gives deeper theoretical understanding of various existing offline RL methods and builds the bridge between behavior regularized and in-sample learning methods in offline RL. We believe that the proposed IVR framework has the potential to scale to other settings, such as online RL and online/offline imitaiton learning.

---------------------------------
## Response to Reviewer PvW6
We thank for minor writing comments you posted, we have made several writing adjustments within the paper to reflect your concerns and suggestions, please see our revision.

>"p.9 "This simulates the situation where the dataset is fewer and has limited state coverage near the target location because the data generation policies maybe not be satisfied and are more determined when they get closer to the target location" what is meant here??"

We have revised this sentence to make it more clear. In Section 5.3, we want to simulate challenges one might encounter when using offline RL algorithms on real-world data. In real-world scenarios, the dataset size may be small or the dataset diversity of some states may be small. For example, in robotic manipulation tasks such as grasping, if the robot is not near the object, it can encounter diverse states by taking different actions and still pick up the object by the end; this is because unless the object breaks, actions taken by the robot are typically reversible, but when the robot grasps the object, its behavior should be more deterministic to ensure successful grasp without damaging or dropping the object.

---------------------------------

## Response to Reviewer u1bC
We thank the reviewer for the thorough and detailed comments.

>"for the methods you compare to, are they using the same hyperparameter for all tasks, or do they also search hyperparameter on each individual dataset and report the best one?"

For the two strongest baselines, IQL and CQL, we initially used scores reported in their paper, except for CQL on AntMaze datasets, as we found the performance can be improved by carefully sweeping the hyperparameter \texttt{min-q-weight} in $[0.5, 1, 2, 5, 10]$.

As you suggested, for a fair comparision, we re-run IQL on all datasets and report the score of IQL by choosing the best score from $\tau$ in $[0.5, 0.6, 0.7, 0.8, 0.9, 0.99]$, using author-provided implementation (https://github.com/ikostrikov/implicit_q_learning). We also re-run CQL on all datasets and report the best score from \texttt{min-q-weight} in $[0.5, 1, 2, 5, 10]$, using a popular PyTorch-version implementation (https://github.com/young-geng/CQL).

We have updated the scores in Table 1 in our paper, we find that the reported scores in CQL and IQL papers are almost the highest one over the searched hyperparameters, SQL still outperforms both CQL and IQL, please refer to the updated paper for details.

>"Similarly, in the case when you compare SQL to CQL and IQL on the noisy and small data regime settings, did you also do a thorough hyperparameter search for IQL and CQL?"

On the noisy and small data regime settings, we did a thorough hyperparameter search for IQL and CQL, we reported the score of IQL by choosing the best score from $\tau$ in $[0.5, 0.6, 0.7, 0.8, 0.9]$, we reported the score of CQL by choosing the best score from \texttt{min-q-weight} in $[0.5, 1, 2, 5, 10]$.

>"Maybe I missed sth but have you provided empirical evidence and figures to demonstrate that your method has a more "sparse" policy than the other competitive methods?"

Thanks for pointing out that, We want to clarify that having a "sparse" policy (only apply sparsity in the policy extraction) is not enough for good performance, we have tried using the policy extraction objective of SQL (i.e., equation (14)) to IQL in our preliminary experiments, but it doesn't get a better policy. 

The sparsity actually helps to learn a better value function (as we initially claimed in Section 4.4), we have added a whole section to elaborate how does sparsity benefit value learning in SQL, in both tabular setting and continuous action setting, please refer to Appendix B for details.

>"What is the final hyperparameter you selected for each individual dataset? (You mentioned the best is selected for each dataset, would be good if you also report the actual numbers)"

We list the per-dataset $\alpha$ of SQL in Table 7. SQL doesn't need to carefully select $\alpha$ as it is robust to a range of different $\alpha$. We can unify $\alpha$ to the following choices: 
- MuJoCo medium and medium-replay: $\alpha=2$
- MuJoCo medium-expert: $\alpha=5$
- AntMaze: $\alpha=0.5$ (except for antmaze-umaze-diverse, we set $\alpha=2$)
- Kitchen: $\alpha=2$

>"What about computation efficiency? How fast in wall-clock speed does your method compare to others?"

We list the runtime of all algorithms in Table 9. SQL has almost the same wall-clock speed as IQL (20m), it can be expected because they only differ in the learning objective of $V$ and $\pi$. However, we want to mention that SQL has faster convergence compared to IQL. 

>"Hyperparameter sensitivity: for example can you provide some results on how performance changes with different alpha values?"

We provide the sensitivity of $\alpha$ in SQL in Table 5 and Table 6, we provide the sensitivity of $\tau in IQL in Table 8. We found that SQL is not much sensitive to a range of different $\alpha$.


>"Will your code be open sourced?"

We will open-source the code and datasets, we have added a "Reproducibilty Statement" section to ensure that.

---------------------------------

## Response to Reviewer HMTv
We would like to thank the reviewer for their in-depth review of our manuscript.

>"Can you provide empirical evidence for whether SQL provide any benefits over IQL in the small-data regime in Section 5.2?"

We have updated Table and add the results of IQL, it is shown that both SQL and IQL outperform CQL while SQL achieves better results than IQL.

>"In-line with the best practices for evaluation, I'd recommend the use of rliable library [1] to validate the claim for SOTA results. Furthermore, standard deviation be reported for all methods in Table 1."

Thank you for your suggestions, we have added the standard deviation of two strongest baselines, IQL and CQL. We also use rliable library to validate the claim for SOTA results of SQL, please see Figure 7 for details.

>"Does the method scale to high-dimensional image-based datasets such as the Atari datasets in RL Unplugged [2]?"

Thanks for pointing out that! We think SQL could scale to those datasets given that CQL had achieved nice results on these datasets and SQL is the "in-sample" version of CQL, which owns more training stability. We will add some experiments about this on the latter version.

>"Kumar et. al (2022) showed degradation in CQL performance with prolonged training (including on the Antmaze datasets) -- would SQL provide any benefit compared over CQL / IQL due to being a principled approach for in-sample learning?"

Thanks for pointing out that! We think both IQL and SQL will provide benefit compared over CQL about the training performance degradation. In the DR3 paper, the authors found that large feature dot products arise when *out-of-sample* actions are used in TD-learning compared to SARSA, despite similar Q-values. Note that both IQL and SQL are SARSA-style learning methods, i.e., using only *in-sample* actions.

>"Are there any guidelines to set the parameter -- given that SQL is theoretically motivated, it would be nice if the hyperparameter tuning is easier / more intuitive than existing methods like CQL / IQL."

To give an intuitive guideline of hyperparameter tuning in SQL, we show the relationship of the normalized score and non-sparsity ratio (i.e., $\mathbb{E}_{(s, a) \sim \mathcal{D}}[\mathds{1} (1+ (Q(s, a) - V(s))/2\alpha >0 )]$) with different $\alpha$ in SQL, in Table 5 and Table 6.

We find the value of non-sparsity ratio is controlled by the hyperparameter $\alpha$, typically a larger $\alpha$ gives less sparsity, sparsity plays an important role in the performance of SQL and we need to choose a proper sparsity ratio to achieve the best performance. The best sparsity ratio depends on the composition of the dataset, for example, the best sparsity ratios in MuJoCo datasets (around 0.1) are always larger than those in AntMaze datasets (around 0.4), this is because AntMaze datasets are kind of multi-task datasets (the start and goal location are different from the current ones), there is a large portion of useless transitions contained so it is reasonable to give those transitions zero weights by using more sparsity.

So the practical guideline of hyperparameter tuning in SQL is that, if the datasets are more diverse and contain a large portion of useless transitions, we should use a lower $\alpha$ to increase the sparsity, otherwise if the datasets are less diverse or require more behavior cloning, we should use a higher $\alpha$ to decrease the sparsity.

>"Can you clarify how Jensen's inequality is applied in Equation 2?"

The regularization term $\mathbb{E}_{\pi} [f(\frac{\pi}{\mu})]$, is equivalent to

$$
\mathbb{E}_{\pi} [f(\frac{\pi}{\mu})] = \mathbb{E}_{\mu} [\frac{\pi}{\mu}f(\frac{\pi}{\mu})]
$$

because $h_f(x) = x f(x)$ is strictly convex, we can apply Jensen's inequality by moving the expectation inside, which has

$$
\mathbb{E}_{\mu} [\frac{\pi}{\mu}f(\frac{\pi}{\mu})] = \mathbb{E}_{\mu} [h_f(\frac{\pi}{\mu})] \geq h_f(\mathbb{E}_{\mu} [\frac{\pi}{\mu}]) = h_f(1) = 1 f(1) = 0
$$

>"The generality of the IVR framework would be seen if another value of  was instantiated to derive in-sample version of another existing method or deriving a new method altogether."

Thanks for pointing out that! We derive another practical new offline RL algorithm, EQL, by applying the reverse KL divergence, we find EQL is an "implicit" version of AWR/AWAC that avoids any out-of-distribution action.

We also test EQL on D4RL benchmark datasets and noisy datasets used in our experiments. It is shown that EQL also has a strong empirical performance. Please see Appendix A for details.

>"...although most methods might turn out to be intractable)."

We discuss about the feasibility of applying other valid choices of $\alpha$-divergence. Please see Appendix A for details.

>"It seems that per-task hyperparameter tuning is done for SQL while the baseline methods' results seem to be copied from prior papers which used the same hyperparameter for a given domain"

Thanks for your suggestions, for a fair comparision, we re-run IQL on all datasets and report the score of IQL by choosing the best score from $\tau$ in $[0.5, 0.6, 0.7, 0.8, 0.9, 0.99]$, using author-provided implementation (https://github.com/ikostrikov/implicit_q_learning). We also re-run CQL on all datasets and report the best score from \texttt{min-q-weight} in $[0.5, 1, 2, 5, 10]$, using a popular PyTorch-version implementation (https://github.com/young-geng/CQL).

We have updated the scores in Table 1 in our paper, we find that the reported scores in CQL and IQL papers are almost the highest one over the searched hyperparameters, SQL still outperforms both CQL and IQL, please refer to the updated paper for details.

>"It's not clear whether "sparsity" plays an important role in performance of SQL "

Thanks for pointing out that. The sparsity actually helps to learn a better value function (as we initially claimed in Section 4.4), we have added a whole section to elaborate how does sparsity benefit value learning in SQL, in both tabular setting and continuous action setting, please refer to Appendix B for details.

>"however the empirical method is highly motivated by the in-sample Q-learning (IQL) method. "

We respectfully disagree with it. Our empirical methods (both SQL and newly added EQL) are derived from the IVR framework, which is obtained by solving a behavior-regularized MDP. Our methods are not motivated by IQL, in fact, we explain why certain choices are used in IQL and also demonstrate its weaknesses.