<div class="title-page">

![UCP logo](docs/assets/ucp_logo.png){width=170px .logo}

<p class="university">UNIVERSITY OF CENTRAL PUNJAB</p>

<p class="faculty">Faculty of Information Technology &amp; Computer Science</p>

<div class="course-block">
<p class="course-name">Introduction to Machine Learning</p>
<p class="course-project">Course Project</p>
</div>

<p class="project-title">Predicting Spotify Track Popularity: A Machine Learning Classification Approach Using Ensemble Methods</p>

<p class="presented-by">Presented by:</p>

<table class="authors">
<tr><td class="author-name">Arslan Ahmed Siddiqui</td><td class="author-roll">L1S23BSCS0254</td></tr>
<tr><td class="author-name">Abdullah Nadeem</td><td class="author-roll">L1S23BSCS0249</td></tr>
</table>

<p class="date">June 2026</p>

</div>

<div class="abstract-section">

## Abstract {.unnumbered .unlisted}

This study addresses the problem of predicting whether a Spotify track will be a "hit" — defined as a popularity score of 50 or above — using only the audio features and genre available at release time. Working with a dataset of approximately 114,000 tracks (89,741 after deduplication and cleaning), we frame the task as binary classification under substantial class imbalance, where hits constitute roughly a quarter of all tracks. The methodology emphasizes a leak-safe preprocessing pipeline: identifier columns are dropped, the interquartile range (IQR) method caps outliers on continuous features only, track genre is one-hot encoded, and SMOTE oversampling is applied strictly within cross-validation folds to avoid information leakage. We compare a Logistic Regression baseline against Random Forest, XGBoost, and LightGBM ensembles, with hyperparameter tuning and 5-fold stratified cross-validation scored on precision-recall AUC. The tuned LightGBM model performed best on the held-out test set, achieving a ROC-AUC of 0.858, a PR-AUC of 0.659, an accuracy of 0.82, and an F1 score of 0.557. Permutation and SHAP analysis reveal that track genre dominates predictive power, and that audio features impose a performance ceiling because popularity is also driven by non-audio market and cultural factors.

</div>
