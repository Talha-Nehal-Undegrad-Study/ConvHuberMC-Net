# Compilation of Senior Project Activities - Electrical Engineering Department

## Introduction
This document summarizes the activities and findings of our senior project focused on deep unfolding for matrix completion. Below is a comprehensive compilation of our coursework, research, experiments, and key insights.

## Coursework and Preliminary Studies
1. **Advanced Digital Signal Processing (ADSP):** Completed Dr. Tahir's online course.
2. **Deep Learning Fundamentals:** Completed an online course by Daniel Burke: [Pytorch for Deep Learning](https://www.learnpytorch.io/).
3. **Literature Review:** Engaged with key texts for foundational knowledge:
   - Read Chapter 4 and the Appendix of [*High-Dimensional Data Analysis with Low-Dimensional Models*](https://book-wright-ma.github.io/Book-WM-20210422.pdf) by John Wright and Yi Ma, covering optimization algorithms and duality theory.

## Research and Experimentation
4. **Paper Implementations:**
   - Replicated findings from Shoaib Bhai's papers on [DUPA-RPCA](https://ieeexplore.ieee.org/document/9906418) and [DUST-RPCA](https://arxiv.org/abs/2211.03184).
5. **Development of ConvMC-Net:**
   - Developed, refined, and documented ConvMC-Net.
   - Ran experiments on synthetic data and compared the results with those from ADMM-Net.
   - Results indicated superior inference capabilities of ConvMC-Net and comparable performance in various test conditions.
   - Detailed documentation is available on [GitHub](https://github.com/Talha-Nehal-Undegrad-Study/convmc-net).
6. **Further Comparisons and Insights:**
   - Compared ConvMC-Net with [$\ell_0$-BCD](https://ieeexplore.ieee.org/abstract/document/9970585), a state-of-the-art method for matrix completion involving Gaussian Mixture Model (GMM) noise, using MATLAB.
   - Identified discrepancies in expected versus actual comparison to be due to different noise assumptions in the objective functions.
7. **Algorithm Exploration:**
   - Switched focus to other algorithms addressing GMM noise, notably [M-Estimation](https://ieeexplore.ieee.org/document/8682657) that utilizes the Huber regression algorithm (hubreg) developed via MM algorithm.
   - Reviewed and unfolded M-estimation based on the [work](https://ieeexplore.ieee.org/document/9231538) of Ollila and Mian concerning sparse learning applications.

## Advanced Experimentation and Unfolding
6. **Unfolding and Optimization:**
   - Initially chose to unfold M-Estimation using the hubreg algorithm described by Ollila and Mian in their paper on sparse learning applications.
   - After unfolding this variant and conducting experiments on synthetic data, we observed unexpected behavior of loss curves, which led us to further investigation of the underlying theory.
   - Upon deeper study of [*Robust Statistics for Signal Processing*](https://www.cambridge.org/core/books/robust-statistics-for-signal-processing/0C1419475504BC0E6C26376185813B6D), we realized that the original hubreg algorithm, which, unlike the previously unfolded variant, had no hyperparameters, was more suitable for our needs.
   - Despite initial reluctance due to its seemingly unlearnable structure, peer discussions convinced us to proceed with unfolding the original hubreg algorithm.
   - Implemented convolution mappings onto the pseudo-inverse matrix as a learnable parameter, which significantly improved our results over the initial approach.
7. **Performance Analysis and Comparisons:**
   - Noticed fluctuations in loss curves for certain combinations of SNR and sampling rate, where validation loss increased significantly for several epochs before decreasing, yet ended higher than it started.
   - Tested different matrix sizes not used in $\ell_0$-BCD experiments and found that M-Estimation and [$\ell_p$-reg](https://ieeexplore.ieee.org/document/8219728) performed better in several combinations.
   - Conducted comparative tests using $\ell_0$-BCD matrix sizes (400x500 and 150x300) for image inpainting. It outperformed others at 400x500, but was comparable to M-estimation and outperformed by $\ell_p$-reg at 150x300.
   - Were suggested by XIao Peng Li that tuning the $\epsilon$ hyperparameter in $\ell_0$-BCD might enhance its performance.
8. **Further Developments and Reflections:**
   - Refined, updated, and implemented MATLAB scripts for various algorithms including $\ell_0$-BCD, M-Estimation, [ORMC](https://ieeexplore.ieee.org/abstract/document/9457222), $\ell_p$-reg (p = 1), and $\ell_p$-ADMM (p = 1), considering these as strong competitors.
   - Aimed to achieve superior results than iterative methods, particularly at higher sampling rates; initial findings showed similar performance at lower rates.
   - Explored enhancements to the unfolding method by learning the pseudo-inverse matrix through a product of two matrices and experimenting with initialization methods (e.g., Xavier) and non-linear activations (e.g., tanh) to improve performance.
   - Observed consistent performance in training and validation loss curves, with ranges between 4e-4 to 3e-4 for training and 7 to 8 for validation which is very 
   suspicious as it seems to imply that learning the psuedo-inverse has somehow made the algorithm indifferent to the noise and sampling rate combinations (note that difference loss metrics were used for training and validation).
   - More details can be found on GitHub regarding our experiments with [M-Estimation](https://github.com/Talha-Nehal-Undegrad-Study/M-estimation-RMC) and its unfolded version, [ConvHuberMC-Net](https://github.com/Talha-Nehal-Undegrad-Study/ConvHuberMC-Net).

## Conclusions and Future Work
9. **Current Findings:**
    - Achieved consistent performance with training and validation loss curves stabilizing within expected ranges.
    - Early results suggest potential for achieving better performance than traditional iterative methods, particularly at higher sampling rates.
10. **Next Steps:**
    - Continue to refine and test ConvMC-Net on image inpainting and additional synthetic data sets.
    - Aim to surpass the performance of iterative versions by addressing identified issues and optimizing algorithm parameters.

## Reflections
- The project provided profound insights into the challenges and complexities of unfolding algorithms for matrix completion in noisy environments.
- In hindsight, despite the fact that this was an entirely new area for us, a few missteps costed us a lot of valuable time and slowed our progress:
    - Upon realizing that ConvMC-Net cannot be fairly compared with ADMM-Net due to difference in model assumptions, it might have been better to consult our peers about whether this is really the case, and, if it is, can we somehow modify ConvMC-Net instead of jumping to a new algorithm.
    - When we decided to unfold M-estimation, we could have asked whether some other algorithm would be more friendly to deep unfolding (we primarily chose M-estimation based on the closeness of its results to those of $\ell_0$-BCD). These lessons will hopefully guide us in our future projects.
- Future projects can include a detailed review of ConvMC-Net followed by our explorative efforts on adapting unfolding techniques to GMM noise problems.