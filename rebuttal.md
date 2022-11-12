Main Contribution
In this paper, we propose a general Implicit Value Regularization framework, which builds the bridge between behavior regularized and in-sample learning methods in offline RL

support constraints

---------------------------------
## Response to Reviewer PvW6
We thank for minor writing comments you posted, we have made several writing adjustments within the paper to reflect your concerns and suggestions, please see our revision.

>"p.9 "This simulates the situation where the dataset is fewer and has limited state coverage near the target location because the data generation policies maybe not be satisfied and are more determined when they get closer to the target location" what is meant here??"

We have revised this sentence to make it more clear. In Section 5.3, we want to simulate challenges one might encounter when using offline RL algorithms on real-world data. In real-world scenarios, the dataset size may be small or the dataset diversity of some states may be small. For example, in robotic manipulation tasks such as grasping, if the robot is not near the object, it can encounter diverse states by taking different actions and still pick up the object by the end; this is because unless the object breaks, actions taken by the robot are typically reversible, but when the robot grasps the object, its behavior should be more deterministic to ensure successful grasp without damaging or dropping the object.

---------------------------------

## Response to Reviewer u1bC
We thank the reviewer for the thorough and detailed comments.

>"About the per-dataset hyperparameter of SQL and hyperparameter of other baselines"

Note that we have only one hyperparameter, while IQL have two.

>"About the per-dataset hyperparameter of SQL and hyperparameter of other baselines"

>"About computation efficiency of SQL"

>"About hyperparameter sensitivity of SQL"


---------------------------------

## Response to Reviewer HMTv
We thank the reviewer for the thorough and detailed comments.

Can you provide empirical evidence for whether SQL provide any benefits over IQL in the small-data regime in Section 5.2?

In-line with the best practices for evaluation, I'd recommend the use of rliable library [1] to validate the claim for SOTA results. Furthermore, standard deviation be reported for all methods in Table 1.

Does the method scale to high-dimensional image-based datasets such as the Atari datasets in RL Unplugged [2]?

Kumar et. al (2022) showed degradation in CQL performance with prolonged training (including on the Antmaze datasets) -- would SQL provide any benefit compared over CQL / IQL due to being a principled approach for in-sample learning?

Are there any guidelines to set the  parameter -- given that SQL is theoretically motivated, it would be nice if the hyperparameter tuning is easier / more intuitive than existing methods like CQL / IQL.

Can you clarify how Jensen's inequality is applied in Equation 2?

The generality of the IVR framework would be seen if another value of  was instantiated to derive in-sample version of another existing method or deriving a new method altogether.


>"...although most methods might turn out to be intractable)."

Note that we have only one hyperparameter, while IQL have two.

>"Error bars are not reported for any of the baselines in Table 1 and 2"

>"The proposed method seems more complex to implement than existing methods such as IQL, while resulting only in marginal gains"

>"It seems that per-task hyperparameter tuning is done for SQL while the baseline methods' results seem to be copied from prior papers which used the same hyperparameter for a given domain"

>"It's not clear whether "sparsity" plays an important role in performance of SQL "

>"however the empirical method is highly motivated by the in-sample Q-learning (IQL) method. "

>"There are not enough details in the appendix to easily replicate the results on D4RL"

Note that we have only one hyperparameter, while IQL have two.