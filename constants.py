# -*- coding: utf-8 -*-
"""
These configurations (CONSTANTS) shouldn't be tuned or modified by the end-user
"""

''' Member Likelihood Segemention on Inference predictions '''

# Members are classified into 3 segments ['High', 'Medium','Low'], based on their likelihood to be Digitally Inclined
# Based on this, Marketing Team can intervene members with
# high & medium likelihood - to achieve their desired goals 
# Please refer Digital Adoption deck for more Insights & Survey Recommendations to Marketing Team

# The threshold criterion is decided by considering a trade-off b/w high Precision@k Recall &
# probability threshold, obtained on test set during development phase (total 13,574 members in Test set)
# The 'k' value is chosen keeping in mind that we're targeting atleast top 5% of total members in Test set

# The ideal selected thresholds are-
# For High segment      :  Precision@15 Recall = 0.8128;  prob_threshold = 0.8225;  Population Covered = 4%
# At 15% Recall, we're targeting considerable proportion of population with a very high precision

# For Medium segment   :  Precision@25 Recall  = 0.6920;  prob_threshold = 0.7169;  Population Covered = 4%
# Members with probability threshold b/w that of 15% Recall & 25% Recall will be marked as Medium,
# beyond which the Precision is dropping significantly. Therefore, the members with the flag of
# low will be considered to be highly probable to turn out as Not-so Digitally Inclined.

# Members with probability score above the threshold of 0.5 will be considered Digitally Inclined, otherwise Not-so Digitally Inclined
# Not-so Digitally Inclined won't be further classified under any targetability segment (high, low, etc.)
# as we're specifically targeting Digital Inclination potentiality in a member
# Therefore, the new members having 'digital_behavior_predicted' as Not-so Digitally Inclined
# will have no value in the 'digitally_inclined_targetability' column in the final inference scoring file
high_prob_threshold      = 0.8225    # prediction probability threshold for a member to be segmented as High
medium_prob_threshold    = 0.7169    # prediction probability  threshold for a member to be segmented as Medium
low_prob_threshold       = 0.5000    # prediction probability  threshold for a member to be segmented as Low

server = 'tl53752.east-us-2.azure'
database='BIE_WH'
username = 'BIE_DEVELOPER'
password = 'BIE2021'

''' Engagement data configurations '''

# Intermediate Portal files
raw_portal_data = '../data/target_data/portal/raw_portal_data.csv'
portal_data = '../data/target_data/portal_data.csv'

# Other engagemnt channels data (find configurations for Portal data in config.py)
subpath = '../data/target_data/'

smt_used_data        = subpath + 'smt_used_data.csv'
app_utilisation_data = subpath + 'app_utilisation_data.csv'
comm_pref_data       = subpath + 'comm_pref_data.csv'
email_data           = subpath + 'email_data.csv'
text_data            = subpath + 'text_data.csv'
wellness_event_data  = subpath + 'wellness_event_data.csv'


''' Engagement data prep configurations '''

features_with_outlier          = 'smt_used'
engagement_score_threshold     = 17
weight_atleast_1_portal_login  = 1
weight_smt_used                = 8
weight_atleast_1_event         = 3
weight_more_than_1_event       = 6
weight_atleast_1_text_clicked  = 5
weight_no_of_text_clicks       = 4
weight_atleast_1_email_opened  = 5
weight_atleast_1_email_clicked = 3
weight_comm_pref_email_text    = 10
weight_comm_pref_mail          = -10
weight_registered_on_app       = 10


''' Columns to filter data used for modeling '''

usecols_IPOP = ['CONSISTENT_MEMBER_ID', 'LEGACY_CLAIM_ID','LEGACY_LINE_ID',
        'ADMIT_DATE', 'DISCHARGE_DATE','FROM_DATE', 'THRU_DATE', 'CLAIM_PAYMENT_STATUS_CODE',
        'PRINCIPAL_DIAG_CODE','SECONDARY_DIAG1_CODE', 'SECONDARY_DIAG2_CODE', 'SECONDARY_DIAG3_CODE',
        'ALLOWED_AMT', 'PAID_AMT', 'COINSURANCE_AMT','COPAY_AMT', 'DEDUCTIBLE_AMT','MARKET_SEGMENT'
]
usecols_Prof = ['CONSISTENT_MEMBER_ID', 'LEGACY_CLAIM_ID','LEGACY_LINE_ID',
        'FROM_DATE', 'THRU_DATE', 'CLAIM_PAYMENT_STATUS_CODE',
        'PRINCIPAL_DIAG_CODE','SECONDARY_DIAG1_CODE', 'SECONDARY_DIAG2_CODE', 'SECONDARY_DIAG3_CODE',
        'ALLOWED_AMT', 'PAID_AMT', 'COINSURANCE_AMT','COPAY_AMT', 'DEDUCTIBLE_AMT','MARKET_SEGMENT']

usecols_RX = ['CONSISTENT_MEMBER_ID', 'LEGACY_MEMBER_ID', 'PHARMACY_NPI_ID', 'PHARMACY_NCPDP_ID', 'LEGACY_CLAIM_ID', 'PRESCRIPTION_ID', 'PRESCRIPTION_FILLED_DATE', 'NDC_CODE', 'PAID_AMT', 'COPAY_AMT',
'DEDUCTIBLE_AMT', 'FILLED_REFILLS_NUM', 'QTY_DISPENSED', 'SUPPLY_DAYS_NUM', 'TIER', 'DAW_CODE', 'COMPOUND_DRUG_CODE', 'PHARMACY_TYPE_CODE', 'MARKET_SEGMENT']

usecols_text_data = ['CCID', 'Sent At Year', 'Clicked']
usecols_comm_pref_data = ['MEME_CK', 'COMMS_PREFERENCE', 'MECM_EFF_DT']


''' Modeling data prep configurations '''
na_imputation_col_list = ['no_of_claims', 'no_of_cases', 'no_of_months_cases_initiated', 'no_of_unique_case_categories']


''' Feature Encoding related configurations '''

encode_features = ['Product_Name']

non_encode_features = ['diag_CIR007', 'diag_END010', 'diag_FAC008', 'diag_FAC016', 'diag_MBD002',
                       'diag_MBD005', 'diag_MUS010', 'diag_MUS011', 'no_of_claims', 'pct_of_denied_claims',
                       'total_claim_amt', 'total_oop_amt', 'avg_time_between_claims', 'pct_of_claims_amt_bucket_low',
                       'pct_of_claims_amt_bucket_med', 'pct_of_claims_amt_bucket_high',
                       'no_of_cases', 'no_of_months_cases_initiated', 'pct_of_case_origin_online',
                       'pct_of_case_origin_offline', 'pct_of_case_origin_physical', 'pct_of_case_origin_others',
                       'avg_case_age', 'no_of_unique_case_categories', 'no_of_plans']


''' Feature Enginnered modeling data path / filenames '''

# SFDC Call Centre data
sfdc_modeling_data = '../data/modeling_data/sfdc_modeling_data.csv'

# Claims data
claims_path = '../data/modeling_data/claims/'
claims_modeling_data = '../data/modeling_data/claims_modeling_data.csv'

temp_ip_path = claims_path+'ip_temp/'
temp_op_path = claims_path+'op_temp/'
temp_prof_path = claims_path+'prof_temp/'
temp_rx_path = claims_path+'rx_temp/'
feature_engg_temp_path = '../data/modeling_data/claims/feature_engg_temp/'

# Member Plan data

member_plan_path = '../data/modeling_data/member_plan/'
product_modeling_data = '../data/modeling_data/product_modeling_data.csv'

product_list = ['healthmate coast to coast coinsurance option', 'bluesolutions hsa', 'anchor plan',
                'bluechip for medicare value', 'vantageblue coinsurance', 'healthmate coast to coast',
                'network blue new england', 'certified dental', 'blue cross dental',
                'lifespan health', 'healthmate for medicare (ppo)', 'blue solutions for hsa direct']


''' Static Data file locations '''
# Static Files are used for some mapping and operations during learner and inference pipeline run
# This files must be present at location : data/static_data/

# Location for various files containing diagnosis codes and mapping used in data preparation and feature engineering
icd_ccsr_codemapping_file = '../data/static_data/ICD10CM_2020_new.xlsx'


''' MISC '''
# Bucket size for extracting member plan data from SQL Server
member_plan_query_bucket = 10000

# Dashboard & Explainer filenames
dashboard_training = '../model_artifacts/training/dashboard/dashboard'
dashboard_inference = '../model_artifacts/inference/dashboard/dashboard'
explainer_training = '../model_artifacts/training/dashboard/explainer'
explainer_inference = '../model_artifacts/inference/dashboard/explainer'

# data columns to upsample, index to be provided as argument
upsample_columns_index = list(range(1,10)) + [16,17,23,24,25]


''' Feature descriptions to be put on explainer dashboard '''

feature_descriptions = {
'Product_Name': 'latest member plan in feature construction period',
'no_of_plans': 'no. of plans a member has been eligible for in feature construction period',
'no_of_claims': 'no. of claims by member in feature construction period',
'pct_of_denied_claims': 'percentage of denied claims out of total no. of claims in feature construction period',
'total_claim_amt': 'total amount paid ($) for all the claims in feature construction period',
'total_oop_amt': 'total out of pocket amount ($) spent by a member',
'avg_time_between_claims': 'Average no of days between consecutive claims in feature construction period', 
'pct_of_claims_amt_bucket_low': 'percentage of claims having paid amount $20 or less',
'pct_of_claims_amt_bucket_med': 'percentage of claims having paid amount between $20-$132',
'pct_of_claims_amt_bucket_high': 'percentage of claims having paid amount more than $132',
'diag_CIR007': 'diagnosis code - essential hypertension', 
'diag_END010': 'diagnosis code - disorders of lipid metabolism', 
'diag_FAC008': 'diagnosis code - neoplasm related encounters',
'diag_FAC016': 'diagnosis code - exposure, encounters, screening or contact with infectious disease',
'diag_MBD002': 'diagnosis code - depressive disorders',
'diag_MBD005': 'diagnosis code - anxiety and fear related disorders', 
'diag_MUS010': 'diagnosis code - musculoskeletal pain, not low back pain',
'diag_MUS011': 'diagnosis code - spondylopathies/spondyloarthropathy (including infective)',
'no_of_cases': 'no. of member-initiated SFDC Call Centre data cases',
'no_of_months_cases_initiated': 'no. of months SFDC cases initiated in',
'pct_of_case_origin_online': 'percentage of cases originated using online mode',
'pct_of_case_origin_offline': 'percentage of cases originated using offline mode', 
'pct_of_case_origin_physical': 'percentage of cases originated using physical mode', 
'pct_of_case_origin_others': 'percentage of cases originated using other modes (Internal)',
'avg_case_age': 'average no. of days in which cases for a member gets resolved, 0 means cases resolved within the same day',
'no_of_unique_case_categories': 'no. of queries cases initiated for'
}
