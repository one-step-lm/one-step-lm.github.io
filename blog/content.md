# How to Generate Text in One Step

> **Subtitle:** Discrete diffusion language models promise fast parallel text generation, but they fall apart with few sampling steps. We show how continuous flows fix this, enabling coherent, high-quality text generation in a single forward pass of the model.

---

## 1. The Dream: Parallel Text Generation

Autoregressive language models like ChatGPT and Claude generate text one token at a time, with the generation of each new token depending on all previous ones. This is a sequential process, and it creates a bottleneck because generating an N-token sentence requires N sequential forward passes through the model.

**Diffusion models** offer an appealing alternative that moves away from the sequential paradigm entirely. Instead of generating left to right, they start from a random sequence and iteratively convert it into real text by processing *all tokens in parallel*. This process is known as denoising, and it typically takes many small steps to produce coherent sentences. The question we'd like to ask is: *is it possible to do this in just a few steps---or even one*?

<!-- VIDEO: videos/flow_text_animation.mp4 (max-width: 360px) -->
> **One-step parallel text generation.** All tokens transition simultaneously from random noise to coherent text in a single forward pass.

We'll show how we can do so using continuous flow-based generative models, in particular by leveraging their distillation into a flow map. 

Before we can understand why continuous flows are the solution, we need to understand a bit more about the limitations of existing discrete diffusion models.

## 2. The Problem with Discrete Diffusion

Discrete diffusion models corrupt text by randomly replacing tokens with other tokens (such as a "mask" token or a uniformly random token) and then learn to reverse this process. They operate on full sequences $\mathbf{y} \in V^L$, where $V$ denotes the vocabulary and $L$ is the sequence length. To go from a noisy sequence at time $s$ to a cleaner one at time $t$, the model needs to estimate the transition probability $p_{t|s}(\mathbf{y}_t | \mathbf{y}_s)$. But this distribution is defined over the full sequence space $V^L$. For a vocabulary of 50,000 tokens and a sequence length of 128, that's $50{,}000^{128}$ possible outputs, so representing this distribution exactly is completely intractable.

To get around this, discrete diffusion models make a **factorization assumption**, approximating the joint transition as a product of independent per-token transitions,

$$p_{t|s}(\mathbf{y}_t | \mathbf{y}_s) \coloneqq p_{t|s}^1(\mathbf{y}_t^1 | \mathbf{y}_s) \cdots p_{t|s}^L(\mathbf{y}_t^L | \mathbf{y}_s),$$

where each factor $p_{t|s}^l$ gives the probability of the $l$-th token. This reduces the problem from exponential to linear in sequence length, making it tractable to learn. Despite its tractability, this approximation assumes that the denoised tokens are *conditionally independent* of each other given the noisy input. Each token is denoised as if the others don't exist.

When you take many small denoising steps, this independence assumption is nearly exact, since in the infinitesimal limit $t \to s$ the factorization becomes tight. Correlations between tokens build up naturally over the course of many iterations. But when you try to take *few* steps, each step must make a large jump, and the factorization breaks down.

> **Key insight:** With few steps, discrete diffusion models cannot capture correlations between tokens. If "New" and "York" are correlated, the model might independently generate "New" for one token and "Diego" for another, producing the nonsensical "New Diego."

<!-- VIDEO: videos/flow_factorization_animation.mp4 (max-width: 720px) -->
> **Factorization error in discrete diffusion.** Consider a toy dataset with two valid phrases: "New York" and "San Diego." With many steps, both discrete diffusion and continuous flows generate valid pairs. With few steps, discrete diffusion independently samples each token, producing spurious mixtures like "New Diego" and "San York." Continuous flows maintain token correlations even with one step.

No amount of better optimization or larger models can fix this, because the factorized form simply cannot represent the correlations that matter for few-step generation. In practice, discrete diffusion models fail in one of two characteristic ways when pushed to few steps: either the output degenerates into random word salad (generative perplexity blows up), or the model hedges by repeating only the most common tokens (entropy collapses). Either way, the practical speedup over autoregressive models is fundamentally limited.

## 3. From Discrete Jumps to Continuous Flows

Our key idea is to abandon the discrete framework entirely and instead work in **continuous space**. Instead of thinking of each token as a discrete symbol to be swapped, we represent it as a point in continuous space using its **one-hot encoding** -- a vector that is 1 in one position and 0 everywhere else.

In this continuous space, we can define a **flow**, which is a smooth transport that carries a cloud of noisy points to the data distribution. Every point in space has a velocity that tells it where to go, and following these velocities from time $t{=}0$ (noise) to $t{=}1$ (data) transforms noise into text.

<!-- VIDEO: videos/flow_matching_animation.mp4 (max-width: 600px) -->
> **Flow matching in continuous space.** A cloud of noisy samples (bottom) is smoothly transported to the target distribution by following a learned velocity field. The path through probability space (top) shows the continuous transformation from noise $\rho_0$ to data $\rho_1$.

In continuous space, the flow operates on *samples* of size $|V|\times L$, naturally avoiding the exponential dependence on the sequence length. That is, there is no need for any factorization assumption: the velocity field naturally captures correlations between all tokens, because it processes the entire sequence as one continuous object.

> **Key insight:** By moving from discrete jumps to continuous flows, we eliminate the need for an independent factorization. The continuous velocity field *jointly* transports all tokens, preserving their correlations at every step.

## 4. Flowing on the Simplex

Because we use one-hot encodings, each token in a vocabulary of $|V|$ words corresponds to a vertex of the **probability simplex**---the space of discrete probability distributions. During the flow, each token starts as a noisy point in Euclidean space, often off the simplex entirely, and gradually drifts toward one of the vertices. At the end, we simply read off the nearest vertex to decode a discrete token.

<!-- VIDEO: videos/simplex_interpolant_animation.mp4 (max-width: 520px) -->
> **Stochastic interpolant on the simplex.** Noisy points (sampled from a Gaussian) are linearly interpolated toward one-hot vertices of the simplex, each representing a word. By $t{=}1$, every point has arrived at a vertex---a discrete token.

This is the elegance of our approach: we train in continuous space using standard flow matching techniques, but the one-hot geometry *automatically* ensures that the output is discrete. No special discretization tricks needed.

We can visualize the learned **velocity field** directly on the simplex. At each point, the velocity tells the flow where to push, and the arrows naturally point toward the correct vertex, guiding every noisy point home.

<!-- VIDEO: videos/velocity_field_animation.mp4 (max-width: 520px) -->
> **Velocity field on the simplex.** The learned velocity field shows the direction each point is pushed at each location. Arrows converge toward the simplex vertices, transporting noisy points to discrete tokens.

## 5. The Denoiser: Classification, Not Regression

A standard flow matching model predicts a **velocity**, which describes a direction to move in continuous space. But for language, we discovered that predicting the velocity directly is a bad idea. The vocabulary is huge ($|V| \approx 50{,}000$), so the velocity lives in a very high-dimensional space, and standard regression losses (like mean squared error) struggle to learn it.

Instead, we reparameterize the model as a **denoiser**. Given a noisy intermediate point $x_t$, the model predicts what the *clean data* $x_1$ should be. Mathematically, the optimal denoiser equals the posterior probability over tokens: $D_t(x) = p(x_1 | x_t)$. This lives on the probability simplex, so we can use a **softmax output** and train with a **cross-entropy loss**. Unlike standard flow matching, which is learned via regression, this means that we learn our model by solving a classification problem.

<!-- VIDEO PAIR -->
<!-- VIDEO: videos/denoiser_animation.mp4 -->
> **Denoiser as posterior.** The denoiser predicts a probability distribution over tokens (shown as blobs at simplex vertices). As $t \to 1$, the distribution concentrates on the correct token.

<!-- VIDEO: videos/classification_animation.mp4 -->
> **Decision regions sharpen over time.** The simplex is partitioned into classification regions. As $t$ increases, the model becomes increasingly confident about which token each noisy point belongs to.
<!-- END VIDEO PAIR -->

This switch from regression to classification makes a significant difference. On our benchmarks, cross-entropy training improves generative perplexity from 120 to 97---a 20% improvement---simply by respecting the discrete geometry of the output space.

> **Key insight:** The optimal denoiser can be viewed as a token-level classifier. Using softmax and cross-entropy instead of square error to learn it dramatically improves language generation quality, because it respects the simplex geometry of discrete tokens.

## 6. Flow Maps: Teleporting to the Answer

So far we have built a flow-based language model (FLM), which generates text by following a velocity field for many steps. We'll show below that this already outperforms discrete diffusion, but we still need to take many steps to accurately resolve the flow. So, then, how do we get to *one* step?

The idea is to learn a **flow map**, which is a function that directly maps a point at any time $s$ to where it *would end up* at a later time $t$, without tracing the intermediate path. Unlike discrete diffusions, this actually exists because our continuous flow is *deterministic*, so that the entire path is determined by the initial Gaussian noise sample. You can think of the flow map as a "teleporter" that says exactly where this noise will end up.

<!-- VIDEO PAIR -->
<!-- VIDEO: videos/rhot_sweep.mp4 -->
> **Continuous flow.** The density smoothly sweeps from $\rho_0$ to $\rho_1$, passing through every intermediate state.

<!-- VIDEO: videos/flowmap_sweep.mp4 -->
> **Flow map.** The flow map "teleports" the density directly between time points, skipping the intermediate states. With a single map $X_{0,1}$, we jump straight from noise to data.
<!-- END VIDEO PAIR -->

### Learning Flow Maps via Distillation

We learn the flow map by **distilling** the pretrained flow model. The key property we exploit is called the **semigroup property**, which states that the flow map from $s$ to $t$ can be decomposed as two consecutive maps $s \to u \to t$. This means we can train the flow map to be self-consistent:

$$X_{s,t}(x) = X_{u,t}(X_{s,u}(x))$$

By enforcing this composition rule during training, the model learns to take larger and larger jumps while maintaining accuracy. Eventually, it can jump from $s{=}0$ all the way to $t{=}1$ in a single step.

<!-- FIGURE: ../figures/semigroup.png (max-width: 600px) -->
> **Semigroup property.** The flow map $X_{s,t}$ (dark arrow) transports directly from $x_s$ to $x_t$. The semigroup condition decomposes this as two consecutive maps: $x_s \to x_u$ then $x_u \to x_t$.

### The Two-Time Denoiser

We typically parameterize the flow map as

$$X_{s,t}(x) = x + (t-s)\,v_{s,t}(x),$$

where $v_{s,t}$ is the average velocity over the interval $[s, t]$. Just as we reparameterized the velocity as a denoiser, we can reparameterize the flow map as a **two-time denoiser**,

$$\delta_{s,t}(x) = x + (1-s)\,v_{s,t}(x),$$

which uses the same average velocity $v_{s,t}$ but takes a single step all the way to $t{=}1$. This object has an important property: it always lies on the probability simplex, regardless of the time interval $(s, t)$. This means we can train it with the cross-entropy loss, inheriting all the benefits of the classification formulation.

When $s = t$, the two-time denoiser reduces to the ordinary denoiser $D_s$. When $t = 1$, $\delta_{s,1}(x) = X_{s,1}(x)$, giving the final prediction directly. Given the two-time denoiser, we can always reconstruct the flow map via

$$X_{s,t}(x) = \frac{1-t}{1-s}\,x + \frac{t-s}{1-s}\,\delta_{s,t}(x).$$

<!-- VIDEO: videos/two_time_denoiser_animation.mp4 (max-width: 600px) -->
> **Two-time denoiser $\delta_{s,t}$.** The two-time denoiser maps a noisy point at time $s$ to a prediction on the simplex, parameterizing the flow map while always remaining on the probability simplex.

### Three Distillation Objectives

There are three equivalent mathematical characterizations of the flow map, each giving rise to a different distillation objective. All three characterize the same object, but they construct the teacher signal in fundamentally different ways. We focus on the **semigroup** objective for our main results, and briefly describe the alternatives for completeness.

**Semigroup.** The idea behind semigroup distillation is progressive composition. We start from the pretrained velocity field, which can only take infinitesimal steps, and train the model to compose pairs of short jumps into single longer jumps. The teacher signal for a jump from $s$ to $t$ is a convex combination of two shorter jumps through an intermediate time $u$,

$$\bar{\delta}_{s,t} = \gamma\,\hat{\delta}_{s,u}(I_s) + (1{-}\gamma)\,\hat{\delta}_{u,t}(X_{s,u}(I_s)).$$

Because the teacher is a convex combination, it always stays on the probability simplex and requires only forward evaluations. Over the course of training, the model learns to make progressively larger jumps, until a single jump from $t{=}0$ to $t{=}1$ produces coherent text.

<!-- VIDEO: videos/progressive_distillation_animation.mp4 (max-width: 450px) -->
> **Semigroup distillation.** The teacher is a convex combination of two shorter jumps on the simplex. No derivatives needed, only forward evaluations.

**Lagrangian.** The Lagrangian teacher transports the sample forward along the flow to time $t$, applies the pretrained denoiser $D_t$ there, and corrects with a time derivative of the student,

$$\bar{\delta}_{s,t} = D_t(X_{s,t}(I_s)) - \tfrac{(1-t)(t-s)}{1-s}\,\partial_t \hat{\delta}_{s,t}(I_s).$$

<!-- VIDEO: videos/lagrangian_distillation_animation.mp4 (max-width: 450px) -->
> **Lagrangian distillation.** The teacher is constructed by transporting $I_s$ forward along the flow, applying the pretrained denoiser at the transported point, and correcting with $\partial_t \hat{\delta}$.

**Eulerian.** The Eulerian teacher evaluates the pretrained denoiser at the starting point and corrects with the Jacobian of the student,

$$\bar{\delta}_{s,t} = D_s(I_s) + \tfrac{t-s}{1-t}\Big((1{-}s)\,\partial_s \hat{\delta}_{s,t}(I_s) + (D_s(I_s) - I_s) \cdot \nabla \hat{\delta}_{s,t}(I_s)\Big).$$

<!-- VIDEO: videos/eulerian_distillation_animation.mp4 (max-width: 450px) -->
> **Eulerian distillation.** The teacher uses the pretrained denoiser at the starting point, corrected by the Jacobian of the student.

> **Why semigroup?** Both the Lagrangian and Eulerian objectives require Jacobian-vector products and may transiently leave the simplex due to the derivative correction terms. The semigroup teacher avoids both issues. Empirically, the semigroup objective achieves **119 generative perplexity** vs. 193 for the Lagrangian. We conjecture that the latter needs additional regularization to keep predictions on the simplex. As a result, we use semigroup distillation for all our main results.

All three objectives also have **self-distillation** variants (Prop. C.11 in the paper), where the model trains from scratch without a pretrained teacher. The diagonal term becomes standard flow matching (cross-entropy to one-hot data), and the off-diagonal teacher uses the model's own predictions. This connects the Eulerian self-distillation to *consistency models* and *MeanFlow*, and the Lagrangian self-distillation to *terminal velocity matching*.

## 7. Putting It All Together

The full pipeline works in two stages:

**Stage 1: Train FLM.** We train a flow-based language model using the denoiser parameterization with the cross-entropy loss. A key ingredient is a **time reparameterization** based on the decoding error rate---the probability that the current noisy state decodes to the wrong token. This redistributes training effort so that each step contributes equally to denoising, which is critical when the vocabulary is large.

### Time Reparameterization: Why It Matters

The decoding error rate $P_e(t)$ measures what fraction of tokens are incorrectly decoded at time $t$. For the linear interpolant, $P_e$ stays high for most of the interval and drops sharply only near $t{=}1$---meaning most of the "action" is concentrated in a tiny time window. Uniform time sampling wastes most of the training budget on times where nothing interesting happens.

The reparameterization $\tau(t) = 1 - \frac{|V|}{|V|-1} P_e(t)$ rescales time so that each unit of $\tau$ corresponds to equal progress in resolving tokens. Compare the two animations below: with uniform time, points (colored red when incorrectly decoded, blue when correct) stay red for most of the animation and snap to blue only at the very end. With reparameterized time, the transition is spread evenly, giving the model uniform signal throughout training.

<!-- VIDEO PAIR -->
<!-- VIDEO: videos/time_reparam_uniform.mp4 -->
> **Uniform time.** Points stay incorrectly decoded (red) for most of the interval and snap to correct (blue) only near $t{=}1$.

<!-- VIDEO: videos/time_reparam_reparam.mp4 -->
> **Reparameterized time.** In $\tau$-space, each particle turns blue at a regular interval, reflecting uniform denoising progress.
<!-- END VIDEO PAIR -->

> **Key insight:** The time reparameterization based on decoding error rate is critical for scaling to large vocabularies ($|V| \approx 50{,}000$). Without it, generative perplexity jumps from 107 to 149---a 40% degradation.

**Stage 2: Distill into FMLM.** We distill FLM into a flow map language model using the two-time denoiser parameterization. Since both the two-time denoiser $\hat{\delta}_{s,t}$ and the semigroup teacher $\bar{\delta}_{s,t}$ lie on the probability simplex, we can train using a KL-based semigroup objective,

$$\mathcal{L}_{\mathsf{KL}}(\hat{\delta}) = \mathbb{E}_{s,u,t}\,\mathbb{E}_{\mathbf{x}_0, \mathbf{x}_1}\Big[\sum_{l=1}^L \mathsf{KL}\big(\bar{\delta}^l_{s,t} \,\Vert\, \hat{\delta}^l_{s,t}(I_s)\big)\Big] + \mathbb{E}_t\,\mathbb{E}_{\mathbf{x}_0, \mathbf{x}_1}\Big[\sum_{l=1}^L \mathsf{KL}\big(\hat{D}^l_t(I_t) \,\Vert\, \hat{\delta}^l_{t,t}(I_t)\big)\Big],$$

where the first term enforces the semigroup condition and the second (diagonal) term ties $\hat{\delta}_{t,t}$ to the pretrained denoiser $\hat{D}_t$. The distilled model can generate text in 1, 2, 4, or 8 steps.

## 8. Results

We evaluate on two standard benchmarks: **LM1B** (One Billion Word) and **OpenWebText** (OWT), measuring generative perplexity (lower is better) and entropy (should match the dataset's entropy).

### Many-step generation

At 1024 steps, FLM already outperforms all discrete diffusion baselines, achieving **96.91** generative perplexity on LM1B (vs. 98.14 for the best baseline, Duo) and **62.23** on OWT (vs. 77.69). This shows that continuous flows are not just useful for speedup---they produce better samples even in the many-step regime.

### Few-step and one-step generation

The real story is in the few-step regime, where FMLM dramatically outperforms distilled discrete diffusion:

<!-- FIGURE: ../figures/combined_few_step_comparison.png -->
> **Few-step generation on LM1B (left) and OpenWebText (right).** FMLM maintains both low generative perplexity and reasonable entropy across all step counts. Discrete baselines either blow up in generative perplexity or collapse in entropy.

> **Headline result:** One-step FMLM achieves **119 generative perplexity** on LM1B, exceeding the **8-step quality of distilled discrete diffusion** models, giving an **8x speedup**. At 4 steps, FMLM reaches **98.76** generative perplexity, outperforming even the best discrete baseline at 1024 steps.

Qualitatively, the difference is even more striking. At one step, discrete diffusion baselines produce either garbled text (generative perplexity >1200) or degenerate text (entropy collapse to 3.79). FMLM generates coherent, grammatically-reasonable sentences while maintaining entropy close to the dataset.

<!-- FIGURE: ../figures/time_vs_ppl.png (max-width: 80%) -->
> **The full picture: quality vs. speed.** FMLM dominates across all step counts, and one-step FMLM is competitive with many-step baselines.

### Why does it work?

Our ablation studies reveal that every piece matters:

- **Denoiser + cross-entropy** vs. velocity + MSE: 97 vs. 3801 generative perplexity. The choice of parameterization is everything.
- **Time reparameterization**: Without it, generative perplexity jumps from 107 to 149. Redistributing training time is critical for large vocabularies.
- **One-hot encoding** outperforms all learned and pretrained embeddings---the simplex geometry is the right inductive bias.
- **Autoguidance**: FLM with guidance reaches **51.62** generative perplexity on LM1B, while discrete baselines collapse under the same guidance strength.

<!-- FIGURE: ../figures/autoguidance_plot.png -->
> **Autoguidance stability.** FLM remains stable across a wide range of guidance scales eta, with autoguidance systematically improving sample quality (generative perplexity down to 51.62 at eta = 50). Discrete baselines (Duo, MDLM) collapse at eta >= 10, with generative perplexity exceeding 1000 and entropy dropping below 3.9.

Beyond unconditional generation, continuous flows also enable **reward-guided generation**. By plugging in classifiers for attributes like topic, grammaticality, sentiment, and safety, FLM and FMLM can steer generation toward desired properties---even at very few steps.

<!-- FIGURE: ../figures/guidance_plot.png -->
> **Reward-guided generation.** Reward scores (top) and generative perplexity (bottom) across sampling steps for four attributes. FLM/FMLM maintains high reward scores with low generative perplexity, while discrete baselines degrade sharply at fewer steps.

## Looking Forward

This work challenges the widely-held assumption that discrete noising processes are necessary for generative modeling over discrete data like text. By showing that continuous flows can outperform discrete diffusion in *both* quality and speed, we open the door to leveraging the rich toolkit of continuous generative modeling---including guidance, inversion, and editing---for language generation.

We believe the most exciting direction is **scaling**: our 170M-parameter model already achieves strong results, and the approach is fully compatible with modern transformer architectures. As these models grow, one-step language generation could become a practical alternative to autoregressive decoding.

---

*For the full technical details, see our paper: [One-step Language Modeling via Continuous Denoising](https://arxiv.org/abs/2602.16813). Code is available on [GitHub](https://github.com/david3684/flm).*
