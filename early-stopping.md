---
title: "Is using a validation set useful for end-to-end learning in robotics?"
thumbnail: /blog/assets/video-encoding/thumbnail.png 
authors:
- user: marinabar
- user: cadene
---

# Is using a validation set useful for end-to-end learning in robotics?

<p>In classical supervised learning, it is common during training to compute metrics like accuracy for classification or mean squared error for regression on a held-out validation set. These metrics are strong indicators of a model capability to generalize to unseen inputs collected in the same context as the training set. Thus, they are used to select the best model checkpoint or to "early-stop" training.</p>
<p>However, in the context of end-to-end imitation learning for real-world robotics, there is no clear consensus among practicioners on the best metrics and practices for using a validation set to select the best checkpoint. This is because the metric that roboticists aim to optimize is the success rate. In other words, the percentage of successful tries in which the robot accomplished the task. It requires running the policy on the robot in the test environement for a long period of time to ensure a low variance caused external factors. For instance, the light conditions can shift, the room layouts can change from day to day, the dynamics of robot motors can change due to usage, etc. More importantly, success rate can not be computed on a validation set. Only the validation loss or other proxy metrics such as the mean squared error in action space can be computed.</p>
<p>Since computing success rate on each model checkpoint is too costly, some practicioners recommend using the validation loss to select the best checkpoint. For instance, <a rel="nofollow" href="https://huggingface.co/blog/m1b/validation-loss-robotics/arxiv.org/pdf/2304.13705">ACT and Aloha</a> authors Zhao et. al. indicate that <em>"at test time, we load the policy that achieves the lowest validation loss and roll it out in the environment"</em>. On the other hand, Stanford <a rel="nofollow" href="https://ai.stanford.edu/blog/robomimic/">Robomimic</a> authors noticed <em>"that the best [policy early stopped on validation loss] is 50 to 100% worse than the best performing policy [when we evaluate all the checkpoints]"</em>, which suggests that selecting the checkpoint with the lowest validation loss does not ensure the highest success rate.</p>
<p>A few hypothesis could explain why low validation loss is not predictive of a high sucess rate. First, there might be a shift in distribution between data collected during training through human teleoperation, and data obtained during evaluation through the policy controlling the robot. This shift can be due to all possible changes in the environment or robot that we previously listed, but also due to slight prediction errors of the policy that accumulate over time and move the robot outside common trajectories. As a result, the inputs seen by the policy during evaluation can be quite different from the ones seen during training. In this context, computing a validation loss might not be helpful since the loss function is used to optimize copying a human demonstrator trajectory, but it does not account for capability to generalize outside of the training distribution. It does not directly optimize the success rate of completing a task in a possibly changing environment.</p>
<p>In this study, we will explore if the validation loss can be used to select the best checkpoint associated with the highest success rate. If it turns out to not be the case, computing a validation loss on a held-out subset of the training set could be useless and may even hurt performance, since training is done on a smaller part of the training set. We will also discuss the alternatives to using a validation loss. Our experiments are conducted in two commonly used simulation environments, PushT and Aloha Transfer Cube, with two different policies, respectively Diffusion and ACT (Action chunking with transformers). Simulation allows us to accurately compute the success rate at every 10K checkpoints, which is challenging in real environments as explained earlier.</p>

## Simulation environment: Pusht

### Experimental Setup

<div style="display: flex; flex-direction: row; align-items: center; justify-content: center;">
  <div style="text-align: center; margin-right: 20px;">
    <img style="width: 150px; max-width: 100%; margin-bottom: 5px;" alt="PushT" src="./Hugging Face – The AI community building the future._files/Ptzl_8QilO3FyCpC0MEK9.png">
    <p style="font-size: 12px; margin: 0;"><strong>Fig. 1:</strong> PushT Environment</p>
  </div>
  <p style="text-align: justify; margin-left: 10px; max-width: 50%;">The diffusion policy was trained on the PushT dataset, with 206 episodes at 10 FPS, yielding a total of 25,650 frames with an average episode duration of 12 seconds.</p>
</div>


<p>We use the same hyperparameters as the authors of Diffusion Policy. We train the policy with three different seeds, then compute the naive mean of each metric. </p>
<p>During training, evaluation is done in simulated environments every 10K steps. We roll out the policy for 50 episodes and calculate the success rate. </p>
<p>Training for 100K steps plus evaluation every 10K steps took about 5 hours on a standard GPU. Running evaluation and calculating success rates is the most costly part, taking on average 15 minutes at each batch rollout.</p>


### Quantitative Results

<p>We compute the Diffusion validation loss on the output of the denoising network. It is the loss between the predicted noise and actual noise used as the training loss. We also compute a more explicit metric to assess the performance of the policy for action prediction, the Mean Squared Error (MSE). We compare <em>N Action Steps</em> worth of predicted actions and ground truth actions. We replicate the process of action selection that is carried out during inference with a queue of observations and actions.</p>
<p>We notice a divergent pattern for validation loss with regards to the success rate, and no correlation between the MSE and the success rate.</p>

<div style="display: flex; justify-content: space-between;">
    <div style="text-align: center; width: 32%;">
        <img style="width: 100%;" alt="PushT Validation Loss" src="./Hugging Face – The AI community building the future._files/8Z3Kmst4NxHI26w7rNjJj.png">
        <p style="font-size: 12px;"><strong>Fig. 2:</strong> PushT Validation Loss</p>
    </div>
    <div style="text-align: center; width: 32%;">
        <img style="width: 100%;" alt="PushT Mean Squared Error" src="./Hugging Face – The AI community building the future._files/VsG0TpxNn4z9fvZn3litg.png">
        <p style="font-size: 12px;"><strong>Fig. 3:</strong> PushT Mean Squared Error</p>
    </div>
    <div style="text-align: center; width: 32%;">
        <img style="width: 100%;" alt="PushT Success Rate" src="./Hugging Face – The AI community building the future._files/nlWGIiAn5zck6WWYiOonx.png">
        <p style="font-size: 12px;"><strong>Fig. 4:</strong> PushT Success Rate</p>
    </div>
</div>


<p>From the first 10,000 steps and until 60,000 steps, validation loss continuously increases, and does not recover to its minimum level by the end of training. In contrast, despite the continuous increase in validation loss, the success rate consistently improves between those steps across all seed runs.</p>
<p>The variations of the mean squared error cannot be used as a reliable point of reference as well. The MSE increases between 40K and 60K steps, but the success rate improves, which contradicts the usual association between lower MSE and higher performance that is seen in classical supervised learning. The MSE decreases between 60K and 70K and increases between 70K and 80K, but for both of those intervals, the success rate falls.</p>
<p>This only shows that no clear signal can be inferred from the changes in the action prediction loss. This holds especially true since the standard deviation (Std) of the MSE Loss at a given step can have the same magnitude as the changes in MSE throughout steps.</p>
<p>We confirm these results by running costly evaluations on 500 episodes to have more samples and decrease variance.
To confirm that there's no correlation between the validation loss and success rate, we evaluate the checkpoints at 20K steps, at 50K steps, and at 90K steps. (<strong>Fig. 5 </strong>) We show the changes relatively to the first column.</p>

<div class="max-w-full overflow-auto">
	<table>
		<thead><tr>
<th><strong>Step</strong></th>
<th><strong>20K steps</strong></th>
<th><strong>50K steps</strong></th>
<th><strong>90K steps</strong></th>
</tr>

		</thead><tbody><tr>
<td>Success Rate (%)</td>
<td>40.47</td>
<td>+55.27%</td>
<td>+25.73%</td>
</tr>
<tr>
<td>Validation Loss</td>
<td>0.0412</td>
<td>+134.57%</td>
<td>+35.94%</td>
</tr>
</tbody>
	</table>
</div>
<p style="font-size: 12px; margin-top:0px;"><strong>Fig. 5:</strong> PushT success rate and denoising validation loss across steps averaged over 3 seeds </p>

<p>The validation losses are more than twice as high after 50K training steps than after 20K training steps, while the success rate improve by over 50% on average. Furthermore, the validation loss decreases between 50K and 90K steps, but the success rate decreases as well.</p>
<p>This suggests limitations of using only validation loss to interpret policy performance.</p>
<p>The variations of the MSE loss are not indicators of evaluation success rate either.</p>
<p>To confirm that there is no correlation between the MSE and success rate, we evaluate the checkpoints at 40K steps, at 60K steps, and at 80K steps. (<strong>Fig. 6 </strong>) We show the changes accross steps relatively to the first column.</p>
<div class="max-w-full overflow-auto">
	<table>
		<thead><tr>
<th><strong>Step</strong></th>
<th><strong>40K steps</strong></th>
<th><strong>60K steps</strong></th>
<th><strong>90K steps</strong></th>
</tr>

		</thead><tbody><tr>
<td>MSE Loss</td>
<td>0.02023</td>
<td>+3.22%</td>
<td>+2.66%</td>
</tr>
<tr>
<td>PC Success (%)</td>
<td>61.08</td>
<td>+2.73%</td>
<td>-17.82%</td>
</tr>
</tbody>
	</table>
</div>
<p style="font-size: 12px; margin-top:0px;"><strong>Fig. 6:</strong> PushT success rate and MSE loss across steps averaged over 3 seeds </p>
<br>

<p>These findings suggest that monitoring metrics alone may not be sufficient for predicting performance in end-to-end imitation learning, nor be used to make informed judgments about stopping training or no.</p>

### Qualitative Results

<p>During training, policy adapts well to handle smooth trajectory planning.</p>
<div style="display: flex; flex-direction: row; align-items: center; justify-content: center;">
    <div style="width: 45%; margin-right: 10px; text-align: center;">
        <img style="width: 100%; max-width: 100%; display: block; margin-left: auto; margin-right: auto;" alt="PushT" src="./Hugging Face – The AI community building the future._files/_0z1E9A2E6THj39IPs-mi.gif">
        <p style="font-size: 12px; margin-bottom: 0;"><strong>Fig. 7:</strong> PushT original example from training set</p>
    </div>
    <div style="width: 45%; margin-left: 10px; text-align: center;">
        <img style="width: 100%; max-width: 100%; display: block; margin-left: auto; margin-right: auto;" alt="PushT" src="./Hugging Face – The AI community building the future._files/UbE2Fr7OJgZt4JQNdWDZB.gif">
        <p style="font-size: 12px; margin-bottom: 0;"><strong>Fig. 8:</strong> PushT rollout episode rendered at a higher resolution</p>
    </div>
</div>

<p>We notice that the policy becomes less jerky with the number of training steps and adapts better to out-of-distribution states. It is also able to plan longer trajectories and predict actions that are more precise in term of distance from current position to next position. </p>
<p><strong>Fig. 10</strong> and <strong>Fig. 11</strong> have the same starting position, but the policy is only able to match the exact T position at the 80K step count. </p>
<div style="display: flex; flex-direction: row; align-items: center; justify-content: center;">
    <div style="width: 30%; margin-right: 10px; text-align: center;">
        <p></p><p><video class="!max-w-full" controls="" src="https://cdn-uploads.huggingface.co/production/uploads/66583df28724def2ab9d231d/3Rg0Kwq2f0GdcxK2xPb6u.mp4"></video></p><p></p>
        <p style="font-size: 11px; margin-bottom: 0;"><strong>Fig. 9:</strong> PushT Diffusion Policy after 20K steps</p>
    </div>
    <div style="width: 30%; margin: 0 10px; text-align: center;">
        <p></p><p><video class="!max-w-full" controls="" src="https://cdn-uploads.huggingface.co/production/uploads/66583df28724def2ab9d231d/OPqlBXgj62ECnft5-tNBB.mp4"></video></p><p></p>
        <p style="font-size: 11px; margin-bottom: 0;"><strong>Fig. 10:</strong> PushT Diffusion Policy after 50K steps</p>
    </div>
    <div style="width: 30%; margin-left: 10px; text-align: center;">
        <p></p><p><video class="!max-w-full" controls="" src="https://cdn-uploads.huggingface.co/production/uploads/66583df28724def2ab9d231d/9fGi52xv9xoF9VF9ZzT25.mp4"></video></p><p></p>
        <p style="font-size: 11px; margin-bottom: 0;"><strong>Fig. 11:</strong> PushT Diffusion Policy after 80K steps</p>
    </div>
</div>

<p>But even at 90K training steps, there are still some failure cases: </p>
<div style="max-width: 30%; margin: auto;">
  <p><video class="!max-w-full" controls="" src="https://cdn-uploads.huggingface.co/production/uploads/66583df28724def2ab9d231d/naSD7Lp1KAKYs4ElH7BEP.mp4"></video></p>
  <p style="font-size: 12px; margin-bottom: 0;"><strong>Fig. 12:</strong> PushT failure case</p>
</div>

## Simulation environment: Aloha Transfer Cube

## Experimental Setup

<div style="display: flex; flex-direction: row; align-items: center;">
  <p style="text-align: justify; margin-right: 10px; max-width: 50%;">In the second simulation, we use the Aloha arms environment on the Transfer-Cube task, with 50 episodes of human-recorded data.
      Each episode consists of 400 frames at 50 FPS, resulting in 8-second episodes captured with a single top-mounted camera.</p>
  <div style="text-align: center; margin-left: 20px;">
    <img style="width: 250px; max-width: 100%; margin-bottom: 5px;" alt="TransferCube" src="./Hugging Face – The AI community building the future._files/E5dxBlMaoCj7pBrCAcds9.png">
    <p style="font-size: 12px; margin: 0;"><strong>Fig. 13:</strong> Aloha Transfer-Cube Environment</p>
  </div>
</div>


<p>We use the same hyperparameters as the authors of ACT. Same as for PushT, we train the policy with three different seeds. </p>
<p>Training for 100K steps + evaluation every 10K steps took about 6 hours on a standard GPU. Running evaluation and calculating success rates is still the most costly part, in this task taking on average 20 minutes at each batch rollout.</p>


### Quantitative Results

<p>In the case of Transfer Cube, we compute the validation loss. We also replicate the process of action selection during inference and compute the MSE on the predicted <em>N Action Steps</em> worth of action every 10,000 steps. Here <em>N Action Steps</em> is equal to 100 and the policy predicts 100 actions at a time. </p>
<p>We notice that while the validation loss plateaus, the success rate continues to grow. We also notice that the variations of the MSE loss are not synchronized with those of the success rate and too variant to be relevant.</p>
<div style="display: flex; justify-content: space-between;">
    <div style="text-align: center; width: 32%;">
        <a rel="nofollow" href="./Hugging Face – The AI community building the future._files/E8vnU3qI1UB4E-ohDrGq-.png">
            <img style="width: 100%;" alt="Transfer Cube Validation Loss" src="./Hugging Face – The AI community building the future._files/E8vnU3qI1UB4E-ohDrGq-.png">
        </a>
        <p style="font-size: 12px;"><strong>Fig. 14:</strong> Transfer Cube Validation Loss</p>
    </div>
    <div style="text-align: center; width: 32%;">
        <a rel="nofollow" href="./Hugging Face – The AI community building the future._files/bnG3ph89WzfE8Y4Mq-J12.png">
            <img style="width: 100%;" alt="Transfer Cube Mean Squared Error" src="./Hugging Face – The AI community building the future._files/bnG3ph89WzfE8Y4Mq-J12.png">
        </a>
        <p style="font-size: 12px;"><strong>Fig. 15:</strong> Transfer Cube Mean Squared Error</p>
    </div>
    <div style="text-align: center; width: 32%;">
        <a rel="nofollow" href="./Hugging Face – The AI community building the future._files/1Er1h9lGBWB7v63yoLw7B.png">
            <img style="width: 100%;" alt="Transfer Cube Success Rate" src="./Hugging Face – The AI community building the future._files/1Er1h9lGBWB7v63yoLw7B.png">
        </a>
        <p style="font-size: 12px;"><strong>Fig. 16:</strong> Transfer Cube Success Rate</p>
    </div>
</div>

<p>The success rate computed during training is highly variant (average of only 50 evaluation episodes) and cannot be conclusive, which is why we run additional evaluations on 500 episodes. </p>
<p>To confirm that there is no correlation between the validation loss and success rate, we calculate the success rate at 30K steps, 70K steps and at 100K steps. (<strong>Fig. 17 </strong>) We show the changes relatively to the first column.</p>
<div class="max-w-full overflow-auto">
	<table>
		<thead><tr>
<th><strong>Step</strong></th>
<th><strong>30K steps</strong></th>
<th><strong>70K steps</strong></th>
<th><strong>100K steps</strong></th>
</tr>

		</thead><tbody><tr>
<td>Success Rate (%)</td>
<td>53.33</td>
<td>+12.94%</td>
<td>+16.67%</td>
</tr>
<tr>
<td>Validation Loss</td>
<td>0.2289</td>
<td>-2.04%</td>
<td>-2.03%</td>
</tr>
</tbody>
	</table>
</div>
<p style="font-size: 12px; margin-top:0px;"><strong>Fig. 17:</strong> Transfer Cube success rate and validation loss averaged over 3 seeds </p>
<br>

<p>So while the validation loss stays roughly the same, or decreases by 2%, the success rate increases by more than 15%. It is challenging to early-stop based on such fine-grained signals; for our task it doesn't appear to be effective.</p>
<p>We run additional evaluations at 50K and 60K steps to confirm that there is no correlation between the MSE loss and the success rate. (<strong>Fig. 18 </strong>) We show the changes relatively to the first column.</p>
<div class="max-w-full overflow-auto">
	<table>
		<thead><tr>
<th><strong>Step</strong></th>
<th><strong>30K steps</strong></th>
<th><strong>50K steps</strong></th>
<th><strong>60K steps</strong></th>
</tr>

		</thead><tbody><tr>
<td>Mean Success Rate</td>
<td>53.33</td>
<td>55.65 (+4.35%)</td>
<td>63.22 (+18.54%)</td>
</tr>
<tr>
<td>MSE Loss</td>
<td>0.8178</td>
<td>0.8153 (-0.31%)</td>
<td>0.8156 (-0.27%)</td>
</tr>
</tbody>
	</table>
</div>
<p style="font-size: 12px; margin-top:0px;"><strong>Fig. 18:</strong> Transfer Cube success rate and MSE loss averaged over 3 seeds </p>
<br>

<p>While the MSE loss does not differ much at every evaluated checkpoint, there is stable improvement in the performance of the model. </p>


### Qualitative Results

<p>We notice that the policy is good at completing unseen trajectories and adapting to out-of-distribution data.</p>
<div style="display: flex; flex-direction: column;">
  <div style="margin-top: 20px; overflow: hidden;">
      <div style="width: 45%; display: inline-block; text-align: center; vertical-align: top; margin-right: 30px;">
          <p style="font-size: 17px; text-align: justify;">Namely, the training set contains little episodes with trajectories readjusted after failing to grasp the cube, or in least cases that readjustment happens in place, by moving the arm horizontally. The closest to readjustment is this type of movement:</p>
      </div>
      <div style="width: 45%; display: inline-block; text-align: center; vertical-align: top;">
          <img style="width: 100%; max-width: 100%; display: block; margin-left: auto; margin-right: auto;" alt="GIF 1" src="./Hugging Face – The AI community building the future._files/UV4Ca5JX5zLjvaOAJOj8h.gif">
          <p style="font-size: 12px; text-align: center;"><strong>Fig. 19:</strong> Example from the training set</p>
      </div>
  </div>
    <div style="margin-top: 20px; text-align: justify;">
        <p style="margin-bottom: 0; font-size: 17px;">But when rolling out the policy upon evaluation, we notice that in many episodes the readjustment of the trajectory happens when the arm is already starting to rise up. This is probably due to the fact that we only train using one top camera, and the policy does not have a good perception of depth, therefore it misjudges cube distances during rollout. The policy often readjusts multiple times, which shows robustness to out-of-distribution data.</p>
    </div>
    <div style="text-align: center; margin-top: 20px;">
        <img style="width: 45%; max-width: 100%; display: block; margin-left: auto; margin-right: auto;" alt="GIF 2" src="./Hugging Face – The AI community building the future._files/05L4eVC5q4bLck0u1U2zd.gif">
        <p style="font-size: 12px; text-align: center;"><strong>Fig. 20:</strong> Episode with multiple trajectory adjustments</p>
    </div> 
    <div style="margin-top: 20px;">
        <div style="width: 45%; margin-right: 30px; margin-top: 40px; text-align: center; display: inline-block; vertical-align: top;">
              <p style="margin-top: 20px; font-size: 17px; text-align: justify;">In some cases, the robot fails to grasp the cube, even after a few attempts.</p>
        </div>
        <div style="width: 45%; text-align: center; display: inline-block; vertical-align: top;">
            <img style="width: 100%; max-width: 100%; display: block; margin-left: auto; margin-right: auto;" alt="GIF 3" src="./Hugging Face – The AI community building the future._files/Evvg6pIrgA1_dITvYS1GN.gif">
            <p style="font-size: 12px; text-align: center;"><strong>Fig. 21:</strong> Failure case </p>
        </div>
    </div>
</div>

<p>While there aren't any informative differences in the losses between 50K and 90K steps, there is improvement in the smoothness of the trajectory:</p>
<div style="display: flex;">
    <div style="width: 45%; margin-bottom: 20px; margin-right:20px;">
        <p style="text-align: justify;"></p><p><video class="!max-w-full" controls="" src="https://cdn-uploads.huggingface.co/production/uploads/66583df28724def2ab9d231d/SoeRbZBz35zZD8bNVKN78.mp4"></video></p><p></p>
        <p style="font-size: 12px; text-align: center; margin-top: 5px;"><strong>Fig 22:</strong> Aloha ACT after 50K steps</p>
    </div>
    <div style="width: 45%; margin-bottom: 20px;">
        <p style="text-align: justify;"></p><p><video class="!max-w-full" controls="" src="https://cdn-uploads.huggingface.co/production/uploads/66583df28724def2ab9d231d/nxN21sIxRbgcxgomrisFh.mp4"></video></p><p></p>
        <p style="font-size: 12px; text-align: center; margin-top: 5px;"><strong>Fig 23:</strong> Aloha ACT after 90K steps</p>
    </div>
</div>

## Conclusion

<p>Our experiments reveal a significant discrepancy between validation loss and task success rate metrics. On our tasks, it is clear that we should not use validation loss to early stop training. This strategy does not ensure the highest success rate. Further studies could be done to exhibit the behaviour of models trained for longer, as it could possibly serve to reduce variance in losses and success rates. In our case, we trained the model until baseline success rate on the given architecture was reached.</p>
<p>In the real world, it is extremely costly to assess the success rate of a given checkpoint with low variance. It surely can not be done at every checkpoint while training. Instead, we advise to run a few evaluation and mainly focus on a qualitative assessment such as the learning of new capabilities and the fluidity of the robot's movements. When no more progress is noticed, the training can be stopped.</p>
<p>For instance, when training <em>PollenRobotics' Reachy2</em> (see <a rel="nofollow" href="https://x.com/RemiCadene/status/1798474252146139595">demo</a>) to grab a cup and place it on a rack, then grab an apple and give to the hand of a person sitting on the opposite side, and rotate back to the initial position ; we noticed that the policy gradually learned more advanced concepts and trajectories:</p>
<ul>
<li>At checkpoint 20k, the robot was only able to grasp the cup, but it was failing to place it on the rack.</li>
<li>At checkpoint 30k, it learned to place it smoothly on the rack, but was not grasping the apple.</li>
<li>At checkpoint 40k, it learned to grasp the apple but was not rotating.</li>
<li>At checkpoint 50k, it learned to rotate and give the apple, but it was not rotating back.</li>
<li>Finally, it learned to rotate back into the desired final position and complete the full trajectory.</li>
</ul>
<p>Doing frequent small qualitative assessments is an efficient method to spot bugs, get a feel of the policy capabilities and stability from one checkpoint to the others, and get inspired on ways to improve it.</p>
<p>In addition, more involved approaches consist in evaluating a policy trained on real data in a realistic simulation, and using the simulation success rate as a proxy to real success rate. These approaches are challenging since they require thorough modeling of robots, environements and tasks with a highly realistic physical engine. As we improve our simulations, scaling these approaches could lead to more efficient training and reduce time/resource costs of evaluation in the real world. In this line of work, we can cite:</p>
<ul>
<li>Li et al. 2024: <a href="https://huggingface.co/papers/2405.05941" rel="nofollow">Evaluating Real-World Robot Manipulation Policies</a></li>
<li>Li et al. 2024: <a href="https://huggingface.co/papers/2406.10788" rel="nofollow">Physically Embodied Gaussian Splatting: A Realtime Correctable World Model for Robotics</a></li>
</ul>

## References

<ul>
<li><a rel="nofollow" href="https://www.ri.cmu.edu/pub_files/2011/4/Ross-AISTATS11-NoRegret.pdf">A Reduction of Imitation Learning and Structured Prediction to No-Regret Online Learning</a></li>
<li><a rel="nofollow" href="https://ai.stanford.edu/blog/robomimic/">Robomimic</a></li>
<li><a rel="nofollow" href="https://arxiv.org/pdf/2108.03298">Evaluating Real-World Robot Manipulation Policies in Simulation</a></li>
<li><a rel="nofollow" href="https://arxiv.org/pdf/2304.13705">Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware</a></li>
<li><a rel="nofollow" href="https://arxiv.org/pdf/2303.04137">Diffusion Policy: Visuomotor Policy Learning via Action Diffusion</a></li>
</ul>