import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt

st.set_page_config(
    # page_title="Finally",
    page_title="CRM - Segmentation",
    page_icon="💳",
    layout="wide"
)

st.title('Segmented Credit Risk Modeling via KMeans Clustering and Validation Metrics')
st.markdown('---')

df = pd.read_excel('Segmentation.xlsx')
st.header('Dataset used for Segmentation')
st.dataframe(df)
st.markdown('---')

#========================================================== EDA ===================================
st.markdown('#### Number of rows')
st.text(df.shape[0])
st.markdown('#### Number of columns')
st.text(df.shape[1])

st.markdown('---')

st.header('Exploratory Data Analysis')
col1, col2 = st.columns(2)
with col1:
    st.markdown('#### Null values in the dataset')
    st.dataframe(df.isnull().sum())
with col2:
    st.markdown('#### Total Nulls in the dataset')
    st.text(df.isnull().sum().sum())

st.markdown('---')

st.header('Min-Mean-Max')
metrics = df.describe().reset_index(names='metric')
metrics = metrics[metrics['metric'].isin(['min','mean','max'])]

st.dataframe(metrics)

st.markdown('---')

st.header('Numerical & Categorical columns')
cat = df.select_dtypes(include = 'object').columns.tolist()
num = df.select_dtypes(include = ['int64','float64']).columns.tolist()
st.markdown('#### Categorical')
st.text(cat)
st.markdown('#### Numerical')
st.text(num)

st.markdown('---')
st.header('Target column')
st.text('Default')
col1, col2 = st.columns(2)
with col1:
    st.dataframe(df.groupby('Default')['Default'].count())


st.markdown('---')
st.header('Dropping ID column')
df = df.drop(columns = ['id'])
st.text("Dropping the column as it doesn't provide any predictive value for the model")

#================================================ OHE ================================================================

st.markdown('---')
st.header('One Hot Encoding')
st.text('Doing One Hot Encoding on the Caegorical columns')
st.text(cat)

from sklearn.preprocessing import OneHotEncoder

enc = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
a = enc.fit_transform(df[cat])
feature_names = enc.get_feature_names_out(df[cat].columns)
a_df = pd.DataFrame(a, columns=feature_names, index=df.index)
df = df.drop(columns=cat)
df = pd.concat([df, a_df], axis=1)

st.dataframe(a_df)

# ========================================== Logistic Regression =====================================================

st.markdown('---')


st.header('Train - Test split')
st.info('Train - Test split = 80% - 20%')

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

X = df.drop(columns = ['Default'])
y = df['Default']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

model_lr = LogisticRegression(max_iter=1000)
model_lr.fit(X_train, y_train)
y_pred_prob_test = model_lr.predict_proba(X_test)[:, 1]


# _-_-_-_-_-_-_-__-_-_-_-_-_-_-_-__-_-_-_-_-_-_-__-_-_-_-_-_-_-__-_-_-_-_-_-_-__-_-_-_-_-_-_-__-_-_-_-_-_-_-_
st.markdown('#### Original Dataset')
st.text(f'Rows : {df.shape[0]} , Columns : {df.shape[1]}')
st.text(df.columns)
with st.expander("Original Dataset Sample"):
        st.dataframe(df.head())

col1, col2 = st.columns(2)
with col1:
    st.markdown('#### Train Dataset')
    st.text(f'Rows : {X_train.shape[0]} , Columns : {X_train.shape[1]}')
    st.text(X_train.columns)
    with st.expander("Train Dataset Sample"):
        st.dataframe(X_train.head())
with col2:
    st.markdown('#### Test Dataset')
    st.text(f'Rows : {X_test.shape[0]} , Columns : {X_test.shape[1]}')
    st.text(X_test.columns)
    with st.expander("Test Dataset Sample"):
        st.dataframe(X_test.head())
# _-_-_-_-_-_-_-__-_-_-_-_-_-_-_-__-_-_-_-_-_-_-__-_-_-_-_-_-_-__-_-_-_-_-_-_-__-_-_-_-_-_-_-__-_-_-_-_-_-_-_

# storing in Y
# df['y_pred_prob'] = model_lr.predict_proba(X)[:,1]

st.markdown('---')
st.header('Logistic Regression')
st.markdown('---')

# ===================== ROC - AUC - Unsegmented Data =========================================================

from sklearn.metrics import roc_auc_score

roc_auc_unseg = roc_auc_score(y_test, y_pred_prob_test)
st.header('ROC - AUC - Unsegmented Test Data')
st.text(roc_auc_unseg)

from sklearn.metrics import roc_curve

fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob_test)

col1, col2  = st.columns(2)
with col1:
    fig, ax = plt.subplots()
    ax.plot(fpr, tpr, label=f"AUC = {roc_auc_unseg:.2f}")
    ax.plot([0, 1], [0, 1], 'k--')
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve")
    ax.legend()
    st.pyplot(fig)

st.markdown('---')

# #================================ KMeans ======================================


st.header('Standard Scaler')

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_train)

st.info('Fitting Standard Scaler on Train Data and Transforming on Test Data on all the columns')

X_scaled_test = scaler.transform(X_test)


col1, col2 = st.columns(2)
with col1:
    st.markdown('#### Train Set Scaled data')
    st.dataframe(X_scaled)
with col2:
    st.markdown('#### Test Set Scaled data')
    st.dataframe(X_scaled_test)

# X_test['y_pred_prob'] = y_pred_prob_test
st.markdown('---')

# #================================ KMeans - 2 ======================================

st.header('KMeans Clustering')
st.info('Fitting KMeans on Train Data and Transforming on Test Data for all the columns and creating clusters')

from sklearn.cluster import KMeans

kmeans2 = KMeans(n_clusters=2 ,random_state=42)
cluster_train = kmeans2.fit_predict(X_scaled)
cluster_test = kmeans2.predict(X_scaled_test)

X_train['cluster_km2'] = cluster_train
X_test['cluster_km2'] = cluster_test


# #================================ KMeans - 3 ======================================

kmeans3 = KMeans(n_clusters=3 ,random_state=42)
cluster_train = kmeans3.fit_predict(X_scaled)
cluster_test = kmeans3.predict(X_scaled_test)

X_train['cluster_km3'] = cluster_train
X_test['cluster_km3'] = cluster_test


#================================ KMeans - 4 ======================================

kmeans4 = KMeans(n_clusters=4 ,random_state=42)
cluster_train = kmeans4.fit_predict(X_scaled)
cluster_test = kmeans4.predict(X_scaled_test)

X_train['cluster_km4'] = cluster_train
X_test['cluster_km4'] = cluster_test

# #================================ KMeans Dataframe ====================================



X_test_with_default = X_test.copy()
X_test_with_default['Default'] = y_test
X_train_with_default = X_train.copy()
X_train_with_default['Default'] = y_train

st.markdown('---')

st.header('Train data KMeans clustering')
col1, col2, col3 = st.columns(3)
with col1:
    st.markdown('#### K=2')
    km2 = X_train_with_default.groupby('cluster_km2')['Default'].agg(['count','mean']).reset_index()
    km2['mean'] = km2['mean']*100
    st.dataframe(km2.rename(columns = {'cluster_km2':'2_Cluster'}))
with col2:
    st.markdown('#### K=3')
    km3 = X_train_with_default.groupby('cluster_km3')['Default'].agg(['count','mean']).reset_index()
    km3['mean'] = km3['mean']*100
    st.dataframe(km3.rename(columns = {'cluster_km3':'3_Cluster'}))
with col3:
    st.markdown('#### K=4')
    km4 = X_train_with_default.groupby('cluster_km4')['Default'].agg(['count','mean']).reset_index()
    km4['mean'] = km4['mean']*100
    st.dataframe(km4.rename(columns = {'cluster_km4':'4_Cluster'}))

st.markdown('---')

st.header('Train data KMeans clustering')
col1, col2, col3 = st.columns(3)
with col1:
    st.markdown('#### K=2')
    km2 = X_test_with_default.groupby('cluster_km2')['Default'].agg(['count','mean']).reset_index()
    km2['mean'] = km2['mean']*100
    st.dataframe(km2)
with col2:
    st.markdown('#### K=3')
    km3 = X_test_with_default.groupby('cluster_km3')['Default'].agg(['count','mean']).reset_index()
    km3['mean'] = km3['mean']*100
    st.dataframe(km3)
with col3:
    st.markdown('#### K=4')
    km4 = X_test_with_default.groupby('cluster_km4')['Default'].agg(['count','mean']).reset_index()
    km4['mean'] = km4['mean']*100
    st.dataframe(km4)


# # ========================== Kmeans Summary ===========================================

st.info("""
Choosing Cluster 2
        
Maximum separation between high and low risk

Easy to interpret and use in scorecards

Avoids splitting low-risk customers into tiny, similar clusters

Adding more clusters (3 or 4) gives almost no new meaningful information.
        
""")
st.markdown('---')

# ========================== Checking for segmentation ================================

st.header('Evaluating Performance of Segmentation')
st.info('1. AUC of a segment in a Segmented Scorecard should be greater than AUC of a Segment in an Unsegmented Scorecard')
st.info('2. Weighted Average AUC of a Segmented Model should be higher than AUC of an Unsegmented Model')
st.info('3. After Segmentation, with same number of Accepts we have Lower Bad rates')
st.markdown('---')


# ========================== Conidtion 1 - Kmeans ================================

# #-------------------------------- 2 clusters == 0 ----------------------------------

st.header('1.  AUC of a segment in a Segmented Scorecard should be greater than AUC of a Segment in an Unsegmented Scorecard')

X_train_km2_0 = X_train[X_train['cluster_km2']==0].drop(columns = ['cluster_km3', 'cluster_km4','cluster_km2'])
y_train_km2_0 = y_train[X_train['cluster_km2'] == 0]

X_test_km2_0 = X_test[X_test['cluster_km2']==0].drop(columns = ['cluster_km3', 'cluster_km4','cluster_km2'])
y_test_km2_0 = y_test[X_test['cluster_km2'] == 0]

# st.text(X_train_km2_0.columns)                
# st.text(X_test_km2_0.columns)

model_km2_0 = LogisticRegression(max_iter=1000)
model_km2_0.fit(X_train_km2_0, y_train_km2_0)
X_test_km2_0_pred = model_km2_0.predict_proba(X_test_km2_0)[:,1]

y_pred_prob_unseg_0 = y_pred_prob_test[X_test['cluster_km2']==0]

st.header('Cluster 0')
st.info('Training Logistic Regression on Train Data - Segment 0 and Predicting on Test Data - Segment 0')

col1, col2 = st.columns(2)
with col1:
    roc_auc_unseg_0 = roc_auc_score(y_test_km2_0,y_pred_prob_unseg_0)
    st.markdown('#### ROC-AUC on Un-Segmented Test Data')
    st.text(roc_auc_unseg_0)
with col2:
    roc_auc_seg_0 = roc_auc_score(y_test_km2_0, X_test_km2_0_pred)
    st.markdown('#### ROC-AUC on Segmented Test Data')
    st.text(roc_auc_seg_0)


col1, col2  = st.columns(2)
with col1:
    fpr, tpr, thresholds = roc_curve(y_test_km2_0,y_pred_prob_unseg_0)
    fig, ax = plt.subplots()
    ax.plot(fpr, tpr, label=f"AUC - Unsegmented Cluster 0 = {roc_auc_unseg_0:.2f}")
    ax.plot([0, 1], [0, 1], 'k--')  # random line
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve")
    ax.legend()
    st.pyplot(fig)
with col2:
    fpr, tpr, thresholds = roc_curve(y_test_km2_0, X_test_km2_0_pred)
    fig, ax = plt.subplots()
    ax.plot(fpr, tpr, label=f"AUC - Segmented Cluster 0 = {roc_auc_seg_0:.2f}")
    ax.plot([0, 1], [0, 1], 'k--')  # random line
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve")
    ax.legend()
    st.pyplot(fig)

st.markdown('---')

# #-------------------------------- 2 clusters == 1  ----------------------------------

X_train_km2_1 = X_train[X_train['cluster_km2']==1].drop(columns = ['cluster_km3', 'cluster_km4','cluster_km2'])
y_train_km2_1 = y_train[X_train['cluster_km2'] == 1]

X_test_km2_1 = X_test[X_test['cluster_km2']==1].drop(columns = ['cluster_km3', 'cluster_km4','cluster_km2'])
y_test_km2_1 = y_test[X_test['cluster_km2'] == 1]

# st.text(X_train_km2_1.columns)                
# st.text(X_test_km2_1.columns)

model_km2_1 = LogisticRegression(max_iter=1000)
model_km2_1.fit(X_train_km2_1, y_train_km2_1)
X_test_km2_1_pred = model_km2_1.predict_proba(X_test_km2_1)[:,1]

y_pred_prob_unseg_1 = y_pred_prob_test[X_test['cluster_km2']==1]


st.header('Cluster 1')
st.info('Training Logistic Regression on Train Data - Segment 0 and Predict on Test Data - Segment 0')

col1, col2 = st.columns(2)
with col1:
    roc_auc_unseg_1 = roc_auc_score(y_test_km2_1,y_pred_prob_unseg_1)
    st.markdown('#### ROC-AUC on Un-Segmented Test Data')
    st.text(roc_auc_unseg_1)
with col2:
    roc_auc_seg_1 = roc_auc_score(y_test_km2_1, X_test_km2_1_pred)
    st.markdown('#### ROC-AUC on Segmented Test Data')
    st.text(roc_auc_seg_1)


col1, col2  = st.columns(2)
with col1:
    fpr, tpr, thresholds = roc_curve(y_test_km2_1,y_pred_prob_unseg_1)
    fig, ax = plt.subplots()
    ax.plot(fpr, tpr, label=f"AUC - Unsegmented Cluster 1 = {roc_auc_unseg_1:.2f}")
    ax.plot([0, 1], [0, 1], 'k--')  # random line
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve")
    ax.legend()
    st.pyplot(fig)
with col2:
    fpr, tpr, thresholds = roc_curve(y_test_km2_1, X_test_km2_1_pred)
    fig, ax = plt.subplots()
    ax.plot(fpr, tpr, label=f"AUC - Segmented Cluster 1 = {roc_auc_seg_1:.2f}")
    ax.plot([0, 1], [0, 1], 'k--')  # random line
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve")
    ax.legend()
    st.pyplot(fig)

st.markdown('---')

# ************************************************************************************************
# # ========================== Conidtion 2 - Kmeans ================================
# ************************************************************************************************

st.header('2. Weighted Average AUC of a Segmented Model should be higher than AUC of an Unsegmented Model')

st.markdown('##### Weighted Avg of Un-Segmented Test Data')
unseg_weight = (X_test_km2_0.shape[0]*roc_auc_unseg_0 + X_test_km2_1.shape[0]*roc_auc_unseg_1) / X_test.shape[0]
st.latex(r"""
\frac{
\text{ROC-Unsegmented}_{0} * \text{Loans}_{0} + 
\text{ROC-Unsegmented}_{1} * \text{Loans}_{1}
}{
\text{Loans}_{\text{total}}
}
""")


st.markdown('##### Weighted Avg of Segmented Test Data')
seg_weight = (X_test_km2_0.shape[0]*roc_auc_seg_0 + X_test_km2_1.shape[0]*roc_auc_seg_1) / X_test.shape[0]
st.latex(r"""
\frac{
\text{ROC-Segmented}_{0} * \text{Loans}_{0} + 
\text{ROC-Segmented}_{1} * \text{Loans}_{1}
}{
\text{Loans}_{\text{total}}
}
""")

st.markdown('---')

st.markdown(f'#### Weighted Avg of Un-Segmented Test Data : **{unseg_weight}**')
st.markdown(f'#### Weighted Avg of Segmented Test Data : **{seg_weight}**')

st.info('Weighted Avg of Segmented Test Data > Weighted Avg of Un-Segmented Test Data')

st.markdown('---')


# ================================= Calculating Scores for Unsegmented DATA ====================================

st.header('Calculating Scores based on Predicted Probability using Logistic Regression')
st.info("""

PDO = 10

Base Score = 500

Offset = Base Score - Factor * Log ( Odds )
        
""")

st.markdown('---')

st.header('Calculating Scores using Predicted Probability on Unsegmented Data')

PDO = 10
base_score = 500
good = (y_train == 0).sum()
bad = (y_train == 1).sum()
odds = good / bad
factor = PDO / np.log(2)
offset = base_score - factor * np.log(odds)
epsilon = 1e-10

pd_unseg = np.clip(y_pred_prob_test, epsilon, 1 - epsilon)

score_unseg = offset + factor * np.log((1 - pd_unseg) / pd_unseg)

X_test['X_test_Unsegment_Score'] = score_unseg

col1, col2 = st.columns(2)
with col1:
    st.dataframe(X_test)
with col2:
    col1, col2 = st.columns(2)
    with col1:
        st.dataframe(X_test['X_test_Unsegment_Score'])

# ================================= Calculating Scores for Test DATA ====================================

st.header('Calculating Scores using Predicted Probability on Test Data')
st.info('''
        Calculating Predicted Probability before Calculating Scores on text data based on the Clusters

        Cluster 0 - Model KM0 ->> Predict Probability after training on Training data of Cluster 0 and predicting on Test data of Cluster 0

        Cluster 1 - Model KM1 ->> Predict Probability after training on Training data of Cluster 1 and predicting on Test data of Cluster 1

        Then calculating Scores on entire Test data using the predicted probabilities
''')

good = (y_train == 0).sum()
bad = (y_train == 1).sum()
odds = good / bad
factor = PDO / np.log(2)
offset = base_score - factor * np.log(odds)
epsilon = 1e-10

X_test_score_km2_0 = X_test[X_test['cluster_km2'] == 0].drop(columns = ['cluster_km3', 'cluster_km4','cluster_km2','X_test_Unsegment_Score'])
X_test_score_km2_1 = X_test[X_test['cluster_km2'] == 1].drop(columns = ['cluster_km3', 'cluster_km4','cluster_km2','X_test_Unsegment_Score'])

pd_c0 = model_km2_0.predict_proba(X_test_score_km2_0)[:, 1]
pd_c1 = model_km2_1.predict_proba(X_test_score_km2_1)[:, 1]

pd_seg = pd.Series(index=X_test.index, dtype=float)
pd_seg.loc[X_test['cluster_km2'] == 0] = pd_c0
pd_seg.loc[X_test['cluster_km2'] == 1] = pd_c1

pd_seg = np.clip(pd_seg, epsilon, 1 - epsilon)
score_seg = offset + factor * np.log((1 - pd_seg) / pd_seg)

X_test['X_test_Score'] = score_seg

col1, col2 = st.columns(2)
with col1:
    st.dataframe(X_test)
with col2:
    col1, col2 = st.columns(2)
    with col1:
        st.dataframe(X_test['X_test_Score'])


st.markdown('---')

# ================================= Calculating Cut OFF Score ====================================

st.header('Score evaluation and Cut off')
st.info('Determining a Cut Off Score using the Un-Segmented data')

X_test_score_cut_off = X_test.copy()
X_test_score_cut_off = X_test_score_cut_off.drop(columns = ['cluster_km3', 'cluster_km4','cluster_km2'])
X_test_score_cut_off['Default'] = y_test


col1 , col2 = st.columns(2)
with col1:
    bin_width = 10
    bins = np.arange(X_test_score_cut_off['X_test_Unsegment_Score'].min() // bin_width * bin_width,X_test_score_cut_off['X_test_Unsegment_Score'].max() + bin_width, bin_width)
    X_test_score_cut_off['X_test_Unsegment_Score_bin'] = pd.cut(X_test_score_cut_off['X_test_Unsegment_Score'], bins)
    score_counts = X_test_score_cut_off.groupby(['X_test_Unsegment_Score_bin', 'Default']).size().unstack(fill_value=0)
    fig, ax = plt.subplots(figsize=(12,8))
    x_labels = score_counts.index.astype(str)
    ax.plot(x_labels, score_counts[0], marker='o', label='Non-Defaulters (0)', color='green')
    ax.plot(x_labels, score_counts[1], marker='o', label='Defaulters (1)', color='red')
    ax.set_title('Score-wise Defaulters vs Non-Defaulters')
    ax.set_xlabel('Score Bins')
    ax.set_ylabel('Number of Customers')
    ax.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    st.pyplot(fig)


st.header("Cut-Off Score Selection")
col1 , col2 = st.columns(2)
with col1:
    X_test_score_cut_off['X_test_Unsegment_Score_bin'] = pd.cut(X_test_score_cut_off['X_test_Unsegment_Score'], bins=10)
    bad_rate_by_bin = X_test_score_cut_off.groupby('X_test_Unsegment_Score_bin')['Default'].mean() * 100

    fig, ax = plt.subplots(figsize=(12,6))
    ax.plot(bad_rate_by_bin.index.astype(str), bad_rate_by_bin, marker='o', color='red')
    ax.set_xlabel("Score Bins")
    ax.set_ylabel("Bad Rate (%)")
    ax.set_title("Bad Rate vs Score")
    plt.xticks(rotation=45)
    plt.grid(True)
    st.pyplot(fig)

with col2:
    st.dataframe(bad_rate_by_bin)


st.info("I selected ~502 as cutoff because it is the point where bad rate drops sharply and stabilizes below ~3%. Below this, risk is significantly higher, so this threshold cleanly separates high-risk and low-risk customers.")

st.markdown('---')

# # ========================== Conidtion 3 - Unsegmented acceptance & Bad Rate ================================

st.header('3. After Segmentation, with same number of Accepts we should have Lower Bad rates')

total_applications = X_test_score_cut_off.shape[0]
approvals = X_test_score_cut_off[X_test_score_cut_off['X_test_Unsegment_Score']>=502].shape[0]
approvals_bad = X_test_score_cut_off[(X_test_score_cut_off['X_test_Unsegment_Score']>=502) & (X_test_score_cut_off['Default']==1)].shape[0]
bad_rate_unsegmented = approvals_bad/ approvals
acceptance_rate_unsegmented = (approvals / total_applications)*100

st.markdown('#### Un-segmented Test Acceptance rate = Approvals / Total Approvals')
st.text('Approvals = Loans greater than and equal to 502 cut off score')
st.text('Total Approvals = Total number of Loans')
col1 , col2,col3, col4 = st.columns(4)
with col1:
    st.info(f'Acceptance Rate : {acceptance_rate_unsegmented}')
st.markdown('#### Un-segmented Test Bad rate = Bad loans in Total Approvals / Total Approvals')
st.text('Bad loans in Total Approvals = Loans greater than and equal to 502 cut off score and that are defaulted')
st.text('Approvals = Loans greater than and equal to 502 cut off score')
col1 , col2,col3, col4 = st.columns(4)
with col1:
    st.info(f'Bad Rate : {bad_rate_unsegmented*100}')


# # ----------- Score cluster 0-------------

st.markdown('#### Segmented Test data - Cluster 0 + Cluster 1')
col1 , col2 = st.columns(2)
with col1:
    st.info(f'To get the same acceptance rate ({acceptance_rate_unsegmented}) for Segmented data ')

# st.text(f"Shape of Cluster 0 : {df[df['cluster_km2']==0].shape[0]}")
# st.text(f"Shape of Cluster 1 : {df[df['cluster_km2']==1].shape[0]}")

X_test_score_cut_off_km = X_test.copy()
X_test_score_cut_off_km['Default'] = y_test
X_test_score_cut_off_km = X_test_score_cut_off_km.drop(columns = ['cluster_km3', 'cluster_km4','cluster_km2'])


total_applications_km0 = X_test_score_cut_off_km.shape[0]
# st.text(f'Shape of Cluster 0 : {total_applications_km0}')
st.text('Minimum and Maximum Score in Test data')
min_score = X_test_score_cut_off_km['X_test_Score'].min()
max_score = X_test_score_cut_off_km['X_test_Score'].max()
st.text(f'Min Score : {min_score}')
st.text(f'Max Score : {max_score}')

new_cutoff = pd.DataFrame()
score_list = []
acceptance_rate_km0_list = []
for i in np.arange(min_score,max_score,1):
    score_list.append(i)
    acceptance_rate_km0_list.append(((X_test_score_cut_off_km[(X_test_score_cut_off_km['X_test_Score']>=i)].shape[0])/total_applications_km0)*100)

new_cutoff['Score'] = score_list
new_cutoff['Acceptance_rate'] = acceptance_rate_km0_list

col1, col2 = st.columns(2)
with col1:
    st.text('Calculating Acceptance Rate for the possible scores')
    st.dataframe(new_cutoff)


st.markdown(f'Score for the almost same Acceptance Rate **{np.round(acceptance_rate_unsegmented)}** in the Segmented Test data **506.06**')
approvals_km0 = X_test_score_cut_off_km[(X_test_score_cut_off_km['X_test_Score']>=506.06)].shape[0]
approvals_bad_km0 = X_test_score_cut_off_km[(X_test_score_cut_off_km['X_test_Score']>=506.06) & (X_test_score_cut_off_km['Default']==1)].shape[0]
bad_rate_segmented_km0 = approvals_bad_km0/ approvals_km0
st.markdown('#### Segmented Test Acceptance rate = Approvals / Total Approvals')
st.text('Approvals = Loans greater than and equal to 506.06 cut off score belonging to Test data using Segmented data Scores')
st.text('Total Approvals = Loans belonging to Cluster 0')
col1, col2 = st.columns(2)
with col1:
    st.info(f'Acceptance Rate : {approvals_km0 / total_applications_km0}')
st.markdown('#### Segmented Test Bad rate = Bad loans in Total Approvals / Total Approvals')
st.text('Bad loans in Total Approvals = Loans with greater than and equal to 506.06 cut off score and that are defaulted belonging to Cluster 0')
st.text('Approvals = Loans greater than and equal to 506.06 cut off score belonging to Test data using Segmented data Scores')
col1, col2 = st.columns(2)
with col1:
    st.info(f'Bad Rate {bad_rate_segmented_km0*100}')
    
st.info(f'With the same acceptance rate, Un-Segmented Data Bad Rate = {np.round(bad_rate_unsegmented*100,2)} & Segmented Data Bad Rate = {np.round(bad_rate_segmented_km0*100,2)}')

st.markdown('---')


# ================================================================================================
# =========================== VALIDATION ================================
# ================================================================================================

st.header('Validation')

st.info('Kolmogorov–Smirnov')
st.info('Rank Ordering')
st.info('Concentration')
st.info('Stability - PSI')

st.markdown('---')

# ================================= KS =================================

st.header('KS')

df_ks = X_test.copy()
df_ks['Default'] = y_test

def calculate_ks_streamlit(df, score_col='X_test_Score', target_col='Default'):
    df = df.sort_values(by=score_col, ascending=True).reset_index(drop=True)

    df['cum_bad'] = (df[target_col] == 1).cumsum() / (df[target_col] == 1).sum()
    df['cum_good'] = (df[target_col] == 0).cumsum() / (df[target_col] == 0).sum()

    df['ks'] = df['cum_bad'] - df['cum_good']
    ks_value = df['ks'].max()

    fig, ax = plt.subplots(figsize=(10,6))
    ax.plot(df['cum_bad'].values, label='Cumulative Bad', color='red')
    ax.plot(df['cum_good'].values, label='Cumulative Good', color='green')
    ax.set_xlabel('Index (sorted by score descending)')
    ax.set_ylabel('Cumulative proportion')
    ax.set_title(f'KS Curve - KS={ks_value:.2f}')
    ax.legend()
    st.pyplot(fig)

    return ks_value



col1, col2 , col3= st.columns(3)
with col1:
    st.markdown("#### KS Calculation for Test Data - Unsegmented")
    ks_value = calculate_ks_streamlit(df_ks, score_col='X_test_Unsegment_Score', target_col='Default')
    col11, col12 , col13= st.columns(3)
    with col12:
        st.info(f"KS value: {ks_value:.2f}")
with col2:
    st.markdown("#### KS Calculation for Test Data - Segmented Cluster 0")
    ks_value = calculate_ks_streamlit(df_ks[df_ks['cluster_km2']==0], score_col='X_test_Score', target_col='Default')
    col11, col12 , col13= st.columns(3)
    with col12:
        st.info(f"KS value: {ks_value:.2f}")
with col3:
    st.markdown("#### KS Calculation for Test Data - Segmented Cluster 1")
    ks_value = calculate_ks_streamlit(df_ks[df_ks['cluster_km2']==1], score_col='X_test_Score', target_col='Default')
    col11, col12 , col13= st.columns(3)
    with col12:
        st.info(f"KS value: {ks_value:.2f}")


ks_table = pd.DataFrame({
    "KS Range": ["< 0.2", "0.2 – 0.4", "0.4 – 0.6", "> 0.6"],
    "Model Performance": ["Poor model", "Moderate model", "Good model", "Excellent model"]
})

st.markdown("### KS Interpretation Table")
st.dataframe(ks_table)


# ====================== Rank Ordering ==================================

st.markdown('---')

st.header('Rank Ordering')

def ks_rank_ordering(df_ks,segment):
    segment_bin = segment + '_bin'
    df_ks[segment_bin] = pd.cut(df_ks[segment], bins=10)
    bin_summary = df_ks.groupby(segment_bin)['Default'].agg(['count','sum']).reset_index()
    bin_summary.rename(columns={'sum':'bad_count'}, inplace=True)
    bin_summary['good_count'] = bin_summary['count'] - bin_summary['bad_count']
    bin_summary['bad_rate'] = bin_summary['bad_count'] / bin_summary['count']
    bin_summary['cum_bad'] = bin_summary['bad_count'].cumsum() / bin_summary['bad_count'].sum()
    bin_summary['cum_good'] = bin_summary['good_count'].cumsum() / bin_summary['good_count'].sum()

    fig, ax = plt.subplots(figsize=(10,6))
    ax.plot(bin_summary[segment_bin].astype(str), bin_summary['cum_bad'], marker='o', label='Cumulative Bad')
    ax.plot(bin_summary[segment_bin].astype(str), bin_summary['cum_good'], marker='o', label='Cumulative Good')
    ax.set_xlabel('Score Bin')
    ax.set_ylabel('Cumulative %')
    ax.set_title('Rank Ordering - Cumulative Good vs Bad')
    ax.legend()
    plt.xticks(rotation=45)
    plt.grid(True)
    st.pyplot(fig)


col1, col2, col3 = st.columns(3)

with col1:
    st.markdown('#### Unsegmented Test data')
    ks_rank_ordering(df_ks,'X_test_Unsegment_Score')

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown('#### Segmented Test data')
    ks_rank_ordering(df_ks,'X_test_Score')
with col2:
    st.markdown('#### Segmented Test data Cluster 0')
    ks_rank_ordering(df_ks[df_ks['cluster_km2']==0],'X_test_Score')
with col3:
    st.markdown('#### Segmented Test data Cluster 1')
    ks_rank_ordering(df_ks[df_ks['cluster_km2']==1],'X_test_Score')

    
st.markdown('---')

# ====================== Concentration ==================================

st.header('Concentration')

concen = X_test.groupby('cluster_km2')['ltv'].count().reset_index()
concen = concen.rename(columns={'cluster_km2': 'Test_Cluster', 'ltv': 'count'})
concen['Concentration %'] = (concen['count']/concen['count'].sum())*100

col1, col2 = st.columns(2)
with col1:
    st.dataframe(concen)

st.info('A single segment should not have > 45% of the total exposure. A segment with more than 45% exposure can be a sign of an imbalance in your segmentation strategy, and it can lead to overfitting, risk concentration, and unfair model predictions.')


# df_ks['X_test_Segment_Score_bin'] = pd.cut(df_ks['X_test_Score'], bins=10)

# bin_summary = df_ks.groupby('X_test_Segment_Score_bin')['Default'].agg(['count', 'sum']).reset_index()
# bin_summary.rename(columns={'sum': 'bad_count'}, inplace=True)
# bin_summary['good_count'] = bin_summary['count'] - bin_summary['bad_count']

# total_bad = bin_summary['bad_count'].sum()

# bin_summary['cum_bad'] = bin_summary['bad_count'].cumsum()
# bin_summary['concentration_ratio'] = bin_summary['cum_bad'] / total_bad

# st.write("Concentration Ratio by Bin:")
# st.dataframe(bin_summary)

# col1, col2, col3 = st.columns(3)

# with col1:
#     fig, ax = plt.subplots(figsize=(10,6))
#     ax.plot(bin_summary['X_test_Segment_Score_bin'].astype(str), bin_summary['concentration_ratio'], marker='o', color='blue')
#     ax.set_xlabel('Score Bin')
#     ax.set_ylabel('Cumulative Concentration Ratio')
#     ax.set_title('Concentration Ratio by Score Bin')
#     plt.xticks(rotation=45)
#     plt.grid(True)
#     st.pyplot(fig)

st.markdown('---')

# ================================= PSI =========================================

st.header('Stability - PSI')

PDO = 10
base_score = 500
good = (y_train == 0).sum()
bad = (y_train == 1).sum()
odds = good / bad
factor = PDO / np.log(2)
offset = base_score - factor * np.log(odds)
epsilon = 1e-10
y_pred_prob_train = model_lr.predict_proba(X_train.drop(columns = ['cluster_km2','cluster_km3','cluster_km4']))[:, 1]
pd_unseg = np.clip(y_pred_prob_train, epsilon, 1 - epsilon)
score_train_unseg = offset + factor * np.log((1 - pd_unseg) / pd_unseg)
X_train['X_train_Unsegment_Score'] = score_train_unseg
# st.dataframe(X_train)

bin_edges = X_train['X_train_Unsegment_Score'].quantile(np.linspace(0, 1, 11)).values
X_train['X_train_Unsegment_Score_bin_psi'] = pd.cut(X_train['X_train_Unsegment_Score'], bins=bin_edges, labels=False, include_lowest=True)

# st.dataframe(X_train.groupby('X_train_Unsegment_Score_bin_psi').size())
# X_train['X_train_Unsegment_Score_bin_psi'] = pd.qcut(X_train['X_train_Unsegment_Score'],10, labels=False, duplicates='drop')
# st.dataframe(X_train.groupby('X_train_Unsegment_Score_bin_psi').size())

X_test['X_test_Unsegment_Score_bin_psi'] = pd.cut(X_test['X_test_Unsegment_Score'], bins=bin_edges, labels=False, include_lowest=True)

X_train_psi = X_train.groupby('X_train_Unsegment_Score_bin_psi').size().reset_index()
X_test_psi = X_test.groupby('X_test_Unsegment_Score_bin_psi').size().reset_index()

# st.dataframe(X_train_psi)
# st.dataframe(X_test_psi)

psi_df = X_train_psi.merge(X_test_psi,left_on='X_train_Unsegment_Score_bin_psi',right_on = 'X_test_Unsegment_Score_bin_psi',how='inner')

psi_df = psi_df.rename(columns = {'0_x':'Dev','0_y':'Val'})

psi_df['Dev_psi'] = (psi_df['Dev']/psi_df['Dev'].sum()*100)
psi_df['Val_psi'] = (psi_df['Val']/psi_df['Val'].sum()*100)

psi_df['Dev-Val'] = (psi_df['Dev_psi'] - psi_df['Val_psi'])
psi_df['log(Dev - Val)'] = np.log(psi_df['Dev_psi'] / psi_df['Val_psi'])
psi_df['psi%'] = psi_df['Dev-Val']*psi_df['log(Dev - Val)']

st.dataframe(psi_df)
st.text(psi_df['psi%'].sum())
st.info('High PSI + stable KS = population shift but model still OK')
st.info('High PSI + KS drop = model breaking')