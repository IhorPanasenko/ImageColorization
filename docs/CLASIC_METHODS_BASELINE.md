# Classic Image Colorization Methods: A Baseline for Comparison

As part of the comparative analysis in this Master's Thesis, two foundational computer vision algorithms are evaluated alongside the Deep Learning (GAN/U-Net) architectures. These classic methods represent the pinnacle of pre-neural-network colorization and serve as a baseline to demonstrate the necessity of semantic scene understanding.

## 1. Optimization-Based Colorization (Scribble-Based)
**Reference:** Levin et al., *"Colorization using Optimization"*, 2004.



### Concept
This is a semi-automatic, local approach. The algorithm requires the user to provide sparse color hints (synthetic scribbles or points) on the grayscale target image. The core assumption is that adjacent pixels with similar luminance levels should have similar colors. The algorithm propagates the provided color hints outwards until it detects a sharp edge (a significant gradient in luminance).

### Mathematical Foundation
The problem is formulated as minimizing a quadratic cost function. For a pixel $r$, the color $U(r)$ should be a weighted average of its neighborhood colors, where the weights $w_{rs}$ are large when luminance is similar:

$J(U) = \sum_{r} \left( U(r) - \sum_{s \in N(r)} w_{rs} U(s) \right)^2$

Minimizing this cost function mathematically translates into solving a massive, sparse system of linear algebraic equations in the form of $Ax = b$. For an image of size $256 \times 256$, the matrix dimensions reach $65536 \times 65536$. Solving this efficiently requires robust numerical methods; the computational complexity here makes optimized numerical solvers (and potentially parallel computing techniques like parallel Gaussian elimination) essential for high-resolution images.

### Pros & Cons
* **Pros:** Mathematically guarantees that color boundaries align perfectly with luminance boundaries. Extremely precise if the hints are accurate.
* **Cons:** Computationally expensive due to the large linear system. It also completely lacks semantic understanding—it cannot color an image without manual hints.

---

## 2. Color Transfer via Statistical Analysis
**Reference:** Welsh et al., *"Transferring Color to Greyscale Images"*, 2002.



### Concept
This is a global, reference-based approach. Instead of manual scribbles, it requires a fully colored "source" image that shares a similar mood or atmosphere with the grayscale "target" image. The algorithm statistically maps the colors from the source to the target.

### Mathematical Foundation
The algorithm operates in the uncorrelated $l\alpha\beta$ color space (conceptually similar to the CIE $L^*a^*b^*$ space used in our deep learning models). For every pixel in the target image, the algorithm searches for the best matching pixel in the source image. The matching is determined by comparing two metrics:
1.  The luminance value ($L$).
2.  The standard deviation of luminance ($\sigma$) within a small local neighborhood (e.g., a $5 \times 5$ window), which serves as a rudimentary texture metric.

Once the best statistical match is found, the chrominance values ($\alpha$ and $\beta$) are simply copied from the source to the target pixel.

### Pros & Cons
* **Pros:** Extremely fast ($O(N)$ complexity depending on the search optimization). Requires no manual drawing, making it easy to automate if a reference dataset is provided.
* **Cons:** "Dumb" matching. Because it only compares luminance and texture statistics, it easily confuses semantically different objects that have similar brightness (e.g., coloring a bright gray sky with the green color of a bright grass patch from the reference image).

---

## Conclusion for Thesis Context
By implementing these methods (using synthetic random points for Levin's method and automated reference pairing for Welsh's method), we can definitively quantify the limitations of classic Computer Vision. It will prove that while mathematical optimization and statistical transfer are powerful, true autonomous colorization requires the deep semantic understanding provided by our proposed Fusion GAN architecture.