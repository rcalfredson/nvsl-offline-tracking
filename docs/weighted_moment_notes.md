**Weighted-moment body estimates, 13 Nov 2026.** Following visual comparison,
selected weighted moments as the working default for ellipse geometry. The
original rectangle-derived estimate remains available with `--shape-fit box`.
This changes the body-size/orientation stage, after foreground segmentation and
candidate acceptance. It does not change the accepted fly center, invent
detections, interpolate missing positions, or smooth trajectories over time.

The previous estimator used the minimum-area rectangle enclosing the selected
contour, then interpreted its center, side lengths and angle as an ellipse.
Extreme foreground pixels can therefore have a large effect on body width and
angle, even when those pixels are only faint extensions of the main body. The
new estimator instead summarizes how the accepted foreground's darkness is
distributed around the same detected center.

**Pixel weights.** For each foreground pixel i within the accepted component,
let B_i be the recording-wide background brightness and I_i its brightness in
the current frame. Define positive darkness and a weight:

\[
D_i=\max(B_i-I_i,0),\qquad q_i=D_i^2.
\]

A pixel 30 grayscale levels darker than background has weight 900; a pixel only
10 levels darker has weight 100. The former has nine times the influence at an
equal distance from the center. This favors the darker interior over weak
peripheral pixels, but it does not prove that every strongly weighted pixel is
anatomical body. Only pixels surviving foreground segmentation inside this
component contribute; the whole rectangular crop is not included.

These shape weights use **raw positive brightness difference squared**, not the
square of the noise-normalized detection score. The noise-normalized threshold
helps decide which pixels enter the foreground; intensity weights then help
estimate the shape of an accepted object. Those are separate stages.

**Second moments about the fixed center.** Keep the accepted rectangle center
c = (c_x,c_y). For a pixel at (x_i,y_i), define offsets u_i = x_i-c_x and
v_i = y_i-c_y. Calculate weighted averages of squared offsets and their product:

\[
M_{xx}=\frac{\sum_i q_i u_i^2}{\sum_i q_i},\qquad
M_{yy}=\frac{\sum_i q_i v_i^2}{\sum_i q_i},\qquad
M_{xy}=\frac{\sum_i q_i u_i v_i}{\sum_i q_i}.
\]

These form a symmetric two-by-two matrix:

\[
M=\begin{pmatrix}M_{xx}&M_{xy}\\M_{xy}&M_{yy}\end{pmatrix}.
\]

M_xx describes horizontal spread, M_yy vertical spread, and M_xy describes
diagonal alignment. Dividing by total weight makes the result a weighted average,
with units of pixels squared. Spread still grows as the body gets larger, but
does not acquire an additional dependence on total weight or a uniform
multiplication of all pixel weights.

This is a second-moment matrix about a **fixed detection center**, not generally
the covariance matrix about the intensity-weighted centroid. If that centroid
is mu and its covariance is C, then M = C + (mu-c)(mu-c)^T. Replacing c with mu
would define a different estimator and would shift the position if mu were
exported. Here c stays unchanged by design; a biased center can consequently
also influence the shape estimate.

**Principal directions and diameters.** The eigenvectors of M identify the
principal directions of spatial spread: its most elongated direction and the
perpendicular direction. Their eigenvalues lambda_max and lambda_min quantify
the spread along those directions. These are the moment estimate's major and
minor axes, without needing to infer either axis from a few extreme contour
points.

To convert spread into ellipse dimensions, use the relation for a uniformly
filled ellipse centered at c. With semiaxes a and b, the moment eigenvalues are
a²/4 and b²/4. For example, this follows by mapping a uniform unit disk to the
ellipse: the disk's average squared x coordinate is 1/4, and scaling x by a
multiplies that moment by a². Therefore:

\[
\text{major diameter}=4\sqrt{\lambda_{\max}},\qquad
\text{minor diameter}=4\sqrt{\lambda_{\min}}.
\]

The factor is four because the square root of the moment gives half a semiaxis,
and the output stores full diameters. Eigenvalues 25 and 4 pixels², for example,
produce diameters 20 and 8 pixels. A real fly has nonuniform intensity and is
not a perfect ellipse, so these are **equivalent moment-based diameters**, not
an exact recovery of its anatomical boundary. The resulting ellipse need not
enclose all foreground pixels.

**Orientation convention.** Export w as minor diameter and h as major diameter.
If the major eigenvector is (v_x,v_y), then its image-plane direction from the
positive x axis is alpha = atan2(v_y,v_x). OpenCV's theta describes the width
axis, so export theta = (alpha_in_degrees - 90) modulo 180. This is equivalent
to the existing convention: a vertical major axis has theta 0, a horizontal
major axis theta 90. The opposite sign of an eigenvector describes the same
body axis, hence the 180-degree periodicity. It does not identify the head.

Angle changes must use the shortest axial difference:

\[
\Delta\theta=((\theta_t-\theta_{t-1}+90)\bmod180)-90.
\]

Thus 179° to 1° is a 2° change, not a 178° rotation. Nearly equal eigenvalues
mean the object is close to circular and its axis direction is poorly
determined, regardless of how precisely an angle is printed.

**Observed effect and what remains fixed.** Over the complete c51 recording,
the moment and box variants have exactly identical x/y values, timestamps,
acceptance decisions and missing masks. Missingness stays 0.851%. Only w, h and
theta change. In the reviewed dark-edge interval [5480,5540), median ellipse
area was about 16% smaller and the 95th-percentile absolute frame-to-frame
axial-angle step fell from 38.2° to 20.8°. There was no temporal smoothing:
every shape was calculated from that frame's pixels. Some other fly/windows
had slightly larger angle steps, and across the entire recording the median
area reduction was larger, about 39–43% depending on fly slot.

These observations support the working-default choice after visual review;
they are not measurements of anatomical accuracy. A tighter ellipse can omit
faint real body parts, a strong reflection can still influence moments, and a
steadier estimate can suppress real posture changes. The next validation is
manual body-outline review and testing additional recordings, especially where
ellipse geometry affects wall-contact measurements. Numerically unresolved
moment estimates retain the rectangle estimate, with the method recorded in
per-frame quality metadata. Original recordings and existing companions remain
unchanged.

Implementation: [`geometry.py`](../src/nvsl_offline/geometry.py), applied after
candidate selection in [`detection.py`](../src/nvsl_offline/detection.py). The
[pilot report](geometry_pilot.md) gives the development windows and full-run
invariance checks.
