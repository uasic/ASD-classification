# %%
# Feature selection
feature_selection_percentage_to_keep = 1

# Cleaning
samples_threshold = 0.90
feature_threshold = 0.80

# Preprocessing
PCA_feature_reduction_bool = False
removing_low_variance_features_bool = False
low_variance_threshold = 0.05
standardization_preprocessing_bool = True
min_max_scaling_preprocessing_bool = False
max_absolute_scaling_preprocessing_bool = False
mean_centering_preprocessing_bool = False
log_normal_transformation_transformation_bool = True
boxcox_transformation_transformation_bool = False
quantile_transformer_transformation_bool = False
mean_imputation_preprocessing_bool = False

# File and model
controls = False
file = "pathways.csv"
model_type = "General-pathways"
output_excel = f"{file}-predictions-{model_type}"


# %%
"""
### Functions
"""

# %%

def pie_charts(dataset):

    import matplotlib.pyplot as plt
    # Ensure the output directory exists
    # Pie chart - diagnosis
    diagnosis = dataset["diagnosis"].value_counts()
    diagnosis.plot(kind='pie', autopct='%1.1f%%', startangle=90, cmap='tab20b')
    plt.title('Diagnosis Distribution')
    plt.ylabel('')

    #plt.tight_layout()
    plt.savefig(f"diagnosis-distribution.png", dpi=350)
    plt.show
    
def merge_dataframes(df1, df2):
    import pandas as pd
    # Concatenate the dataframes along the index
    merged_df = pd.concat([df1, df2], axis=0)
    # Group by the index and aggregate the values
    merged_df = merged_df.groupby(merged_df.index).first()
    return merged_df

def run_permanova(dataset, feature, to_drop):
    import pandas as pd
    from scipy.spatial.distance import pdist, squareform
    from skbio.stats.distance import DistanceMatrix
    from skbio.stats.distance import permanova

    # Step 1: Prepare data
    permanova_df = dataset.copy()
    permanova_df.columns = permanova_df.columns.astype(str)

    if feature in to_drop:
        to_drop = [col for col in to_drop if col != feature]

    permanova_df = permanova_df.drop(columns=[col for col in to_drop if col in permanova_df.columns], errors='ignore')
    
    if feature not in permanova_df.columns:
        print(f"\nFeature '{feature}' not found after dropping columns. Skipping PERMANOVA for this feature.")
        return None, None
    
    permanova_df = permanova_df.dropna(subset=[feature])
    
    # Step 2: Factorize the grouping variable
    permanova_df['labels'], _ = pd.factorize(permanova_df[feature])
    permanova_df = permanova_df.drop(columns=[feature])

    # Step 3: Independent variables and DistanceMatrix
    independent_variables = permanova_df.drop(columns=['labels']).astype(float)
    ids = independent_variables.index.astype(str)
    distance_matrix = squareform(pdist(independent_variables, metric='euclidean'))
    distance_matrix_object = DistanceMatrix(distance_matrix, ids=ids)

    # Step 4: Ensure grouping has matching index (string type)
    grouping_values = permanova_df.loc[independent_variables.index, 'labels'].values
    grouping = pd.Series(grouping_values, index=ids, name='labels')

    # Step 5: Run PERMANOVA
    result = permanova(distance_matrix_object, grouping=grouping)

    print(f"PERMANOVA results for {feature}:")
    print("F-statistic:", result['test statistic'])
    print("p-value:", result['p-value'])

    return str(result['p-value']), str(result['test statistic'])

def pls_variance_explained(data, factor_variable, to_drop):
    import numpy as np
    from sklearn.preprocessing import OneHotEncoder, StandardScaler
    from sklearn.cross_decomposition import PLSRegression
    import pandas as pd

    print(f"Calculating explained variability for factor: '{factor_variable}'...")

    # Prepare response variables (exclude metadata, including factor_variable itself)
    to_drop_extended = to_drop + [factor_variable]
    response_variables = data.drop(columns=to_drop_extended, errors='ignore').select_dtypes(include=[np.number]).columns.tolist()

    if not response_variables:
        raise ValueError("No numeric response variables found after excluding metadata.")

    Y = data[response_variables].values

    # Prepare predictor X (diagnosis/factor_variable)
    X = data[[factor_variable]].astype(str)
    
    # Optional: replace TD/ASD with 0/1 if desired
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    X_encoded = le.fit_transform(X[factor_variable]).reshape(-1, 1)

    # Scale Y
    scaler = StandardScaler()
    Y_scaled = scaler.fit_transform(Y)

    # PLS regression
    n_components = min(X_encoded.shape[1], Y.shape[1], 2)
    pls = PLSRegression(n_components=n_components, max_iter=10000)
    pls.fit(X_encoded, Y_scaled)
    Y_pred = pls.predict(X_encoded)

    # Variance explained
    total_variance_Y = np.sum(np.var(Y_scaled, axis=0, ddof=1))
    explained_variance = np.sum(np.var(Y_pred, axis=0, ddof=1))
    percentage_explained = (explained_variance / total_variance_Y) * 100

    print(f"Percentage of variance explained by '{factor_variable}': {percentage_explained:.2f}%")
    return percentage_explained

def samples_selection(df, threshold):
    print(f"Samples with more than {threshold*100} percentage of zero values will be removed.")
    # Define the threshold for zero values in a row (90%)
    threshold = threshold
    # Calculate the percentage of zeros in each row
    zero_percentage = (df == 0).mean(axis=1)
    # Filter out rows with more than 90% zeros
    df = df[zero_percentage < threshold]
    return df

def nan2zero(dataset):
    numeric_columns = dataset.select_dtypes(include=['number']).columns
    dataset[numeric_columns] = dataset[numeric_columns].fillna(0)
    dataset = dataset.fillna("Unknown")
    return dataset

def data_cleaning(dataset, threshold, metadata):
    print(f"Features with more than {threshold*100} percentage of zero values will be removed.")
    # Replacing negative values with 0
    print("Replacing NaN and negative values with 0.0 ...")
    numeric_columns = dataset.select_dtypes(include=['number']).columns
    dataset[numeric_columns] = dataset[numeric_columns].fillna(0).clip(lower=0)
    columns_with_nan = dataset.columns[dataset.isna().any()].tolist()
 
    try:
        for element in columns_with_nan:
            if element not in metadata:
                 dataset[element] = dataset[element].fillna(0)
    except:pass
    
    columns_to_drop = dataset.columns[(dataset == 0).mean() > threshold]
    dataset = dataset.drop(columns=columns_to_drop)
    columns_with_nan = dataset.columns[dataset.isna().any()].tolist()
    # Eliminating columns with zero values only.
    dataset = dataset.loc[:, (dataset != 0).any(axis=0)]
    return dataset

def reduce_dimensionality_pca(df, to_drop, target_column, percentage):
    import pandas as pd
    import numpy as np
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    # Step 1: Separate features and target
    X = df.drop(columns=[col for col in to_drop if col in df.columns], errors='ignore')
    X = df.drop(target_column, axis=1)
    y = df[target_column]

    # Step 2: Scale the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Step 3: Determine number of components (x% of original features)
    rows, columns = X_scaled.shape
    n_components = rows*percentage
    n_components = int(n_components)
    print("Number of components: ", n_components)
    print(f"Reducing to {n_components} principal components (10% of original {rows} features).")

    # Step 4: Apply PCA
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X_scaled)

    # Step 5: Return the reduced data as a DataFrame, along with the target
    pca_columns = [f'PC{i+1}' for i in range(n_components)]
    X_pca_df = pd.DataFrame(X_pca, columns=pca_columns, index=df.index)
    reduced_df = merge_dataframes(y, X_pca_df)

    return reduced_df, pca, scaler

def standardization(df):
    import pandas as pd
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    numeric_cols = df.select_dtypes(include=['number']).columns
    # Fit and transform only numeric columns
    df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
    numeric_df = df[numeric_cols]
    other_df = df.drop(columns=numeric_cols)
    scaled_numeric_df = pd.DataFrame(scaler.fit_transform(numeric_df), columns=numeric_cols, index=numeric_df.index)
    #display(scaled_numeric_df)
    # Combine back
    df_scaled = merge_dataframes(other_df, scaled_numeric_df)
    return df_scaled

def min_max_scaling(df):
    import pandas as pd
    from sklearn.preprocessing import MinMaxScaler
    
    scaler = MinMaxScaler(feature_range=(0, 1))
    numeric_cols = df.select_dtypes(include=['number']).columns
    
    numeric_df = df[numeric_cols]
    other_df = df.drop(columns=numeric_cols)
    
    scaled_numeric_df = pd.DataFrame(scaler.fit_transform(numeric_df), columns=numeric_cols, index=numeric_df.index)
    df_scaled = merge_dataframes(other_df, scaled_numeric_df)    
    return df_scaled

def max_absolute_scaling(df):
    import numpy as np
    
    numeric_cols = df.select_dtypes(include=['number']).columns
    numeric_df = df[numeric_cols].copy()
    other_df = df.drop(columns=numeric_cols)
    
    for col in numeric_cols:
        max_abs = np.max(np.abs(numeric_df[col]))
        if max_abs != 0:
            numeric_df[col] = numeric_df[col] / max_abs
        else:
            # avoid division by zero
            numeric_df[col] = numeric_df[col]
    
    df_scaled = merge_dataframes(other_df, numeric_df)    
    return df_scaled
    
def mean_centering(df):
    import pandas as pd
    
    numeric_cols = df.select_dtypes(include=['number']).columns
    numeric_df = df[numeric_cols].copy()
    other_df = df.drop(columns=numeric_cols)
    
    numeric_df = numeric_df - numeric_df.mean()
    
    df_scaled = merge_dataframes(other_df, numeric_df)
    return df_scaled

def log_normal_transformation(df):
    import pandas as pd
    import numpy as np

    numeric_cols = df.select_dtypes(include=['number']).columns
    numeric_df = df[numeric_cols].copy()
    other_df = df.drop(columns=numeric_cols)

    # Apply log(1 + x) transformation to handle zeros
    numeric_df = np.log1p(numeric_df)

    df_transformed = merge_dataframes(other_df, numeric_df)
    return df_transformed

def boxcox_transformation(df):
    import pandas as pd
    from sklearn.preprocessing import PowerTransformer

    numeric_cols = df.select_dtypes(include=['number']).columns
    numeric_df = df[numeric_cols].copy()
    other_df = df.drop(columns=numeric_cols)

    epsilon = 1e-6
    numeric_df = numeric_df + epsilon
    
    pt = PowerTransformer(method='box-cox')
    transformed = pt.fit_transform(numeric_df)
    numeric_df = pd.DataFrame(transformed, columns=numeric_cols, index=numeric_df.index)

    df_transformed = merge_dataframes(other_df, numeric_df)
    
    return df_transformed
    
def quantile_transformer(df, n_quantiles, output_distribution='uniform'):
    import pandas as pd
    from sklearn.preprocessing import QuantileTransformer

    numeric_cols = df.select_dtypes(include=['number']).columns
    numeric_df = df[numeric_cols].copy()
    other_df = df.drop(columns=numeric_cols)

    qt = QuantileTransformer(output_distribution=output_distribution, random_state=42)
    transformed = qt.fit_transform(numeric_df)
    numeric_df = pd.DataFrame(transformed, columns=numeric_cols, index=numeric_df.index)

    df_transformed = merge_dataframes(other_df, numeric_df)
    return df_transformed
    
def mean_imputation(df):
    numeric_cols = df.select_dtypes(include=['number']).columns
    df_imputed = df.copy()
    for col in numeric_cols:
        mean_val = df_imputed[col].mean()
        df_imputed[col] = df_imputed[col].fillna(mean_val)
        display(df_imputed)
    return df_imputed

def remove_low_variance_columns(df, threshold):
    # Identify numeric columns
    numeric_cols = df.select_dtypes(include=['number']).columns
    # Identify low variance columns (constant or near-constant)
    low_variance_cols = [col for col in numeric_cols if df[col].var() < threshold]
    if low_variance_cols:
        print(f"Removing columns with variance below {threshold}: {low_variance_cols}")
    df_reduced = df.drop(columns=low_variance_cols)
    return df_reduced

def model_performance(y_test, y_pred, y_prob):
    import pandas as pd
    from sklearn.metrics import (confusion_matrix, ConfusionMatrixDisplay, roc_auc_score, 
                                 roc_curve, accuracy_score, precision_score, recall_score, f1_score)
    import matplotlib.pyplot as plt
    import numpy as np

    # Binary encoding
    label_map = {'TD': 0, 'ASD': 1}
    y_test_bin = pd.Series(y_test).map(label_map).astype(int)
    y_pred_bin = pd.Series(y_pred).map(label_map).astype(int)

    print("Label encoding: TD -> 0, ASD -> 1")

    # AUC and ROC
    if len(np.unique(y_test_bin)) == 2:
        auc_score = roc_auc_score(y_test_bin, y_prob)

        fpr, tpr, _ = roc_curve(y_test_bin, y_prob)
        plt.figure()
        plt.plot(fpr, tpr, label=f'AUC = {auc_score:.2f}')
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.legend(loc='lower right')
        plt.grid(True)
        plt.show()
    else:
        print("AUC calculation for multiclass requires additional parameters.")
        auc_score = None

    # Confusion matrix
    cm = confusion_matrix(y_test_bin, y_pred_bin, labels=[0, 1])
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["TD", "ASD"])
    disp.plot(cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.grid(False)
    plt.show()

    # Classification metrics
    accuracy = accuracy_score(y_test_bin, y_pred_bin)
    precision = precision_score(y_test_bin, y_pred_bin, pos_label=1)
    recall = recall_score(y_test_bin, y_pred_bin, pos_label=1)
    f1 = f1_score(y_test_bin, y_pred_bin, pos_label=1)

    tn, fp, fn, tp = cm.ravel()
    fpr = fp / (fp + tn)
    fnr = fn / (tp + fn)

    print(f"AUC: {auc_score:.2f}" if auc_score else "AUC: Not applicable")
    print(f"Accuracy : {accuracy:.2f}")
    print(f"Precision: {precision:.2f}")
    print(f"Recall   : {recall:.2f}")
    print(f"F1 Score : {f1:.2f}")
    print(f"True Positive : {tp}")
    print(f"True Negative : {tn}")
    print(f"False Positive Rate: {fpr:.2f}")
    print(f"False Negative Rate: {fnr:.2f}")

    return auc_score, accuracy, precision, recall, f1, tn, tp, fn, fp, fpr, fnr

def random_forest_feature_selection(X, y, df, feature_percentage, file):
    from sklearn.ensemble import RandomForestClassifier
    import numpy as np
    import pandas as pd

    rf = RandomForestClassifier(random_state=42)
    rf.fit(X, y)

    # Get feature importances
    importances = rf.feature_importances_
    feature_names = X.columns
    feature_importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
    #display(feature_importance_df.sort_values(by='Importance', ascending=False))
    
    indices = np.argsort(importances)[::-1]

    # Select top k features
    num_columns = len(df.columns)
    print("Number of columns in original dataset before feature selection: ", num_columns)
    k = round(num_columns*feature_percentage)
    top_features = indices[:k]
    X_new = X.iloc[:, top_features]  # Use iloc for DataFrames
    num_columns_new = len(X_new.columns)
    print("Number of columns in dataset after feature selection: ", num_columns_new)
    #display(X_new)
    X_new.to_csv(f"selected-features-{file}")
    return X_new

def prediction(X_train, X_test, y_train, y_test, data, metadata, cleaned_data):
    import pandas as pd
    import numpy as np
    from sklearn.pipeline import Pipeline
    from sklearn.compose import ColumnTransformer
    from sklearn.preprocessing import OneHotEncoder, LabelBinarizer
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.base import BaseEstimator, TransformerMixin

    classifier = RandomForestClassifier(
                    n_estimators=1000,
                    max_depth=10,              # limit tree depth
                    min_samples_split=5,       # don't split if < 5 samples
                    min_samples_leaf=3,        # each leaf must have ≥ 3 samples
                    max_features='sqrt',       # fewer features = less overfitting
                    random_state=42
                )

    # Custom preprocessing wrapper
    class CustomPreprocessing(BaseEstimator, TransformerMixin):
        def __init__(self, data, metadata, cleaned_data):
            self.data = data
            self.metadata = metadata
            self.cleaned_data = cleaned_data
            self.feature_names = None

        def fit(self, X, y=None):
            self.y = y  # Save y so it's available in transform
            # Save column names if input is a DataFrame
            if isinstance(X, pd.DataFrame):
                self.feature_names = X.columns
            return self

        def transform(self, X):
            # Convert to DataFrame if X is a NumPy array
            if isinstance(X, np.ndarray):
                if self.feature_names is not None:
                    X = pd.DataFrame(X, columns=self.feature_names)
                else:
                    X = pd.DataFrame(X)  # fallback: unnamed columns

            X_preprocessed = preprocessing(
                X_encoded=X,
                y_encoded=self.y,
                data=self.data,
                metadata=self.metadata,
                cleaned_data=self.cleaned_data
            )
            return X_preprocessed

    # Identify categorical and numeric columns
    categorical_cols = X_train.select_dtypes(exclude=['int64', 'float64', 'int32', 'float32']).columns.tolist()
    numeric_cols = X_train.select_dtypes(include=['int64', 'float64', 'int32', 'float32']).columns.tolist()
    
    # Column transformer
    column_transformer = ColumnTransformer([
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols),
        ('num', 'passthrough', numeric_cols)
    ])

    # Pipeline
    custom_preprocessor = CustomPreprocessing(data=data, metadata=metadata, cleaned_data=cleaned_data)
    pipeline = Pipeline([
        ('main_preprocessing', custom_preprocessor),  # run your custom preprocessing first on raw data
        ('preprocessor', column_transformer),         # then encode categorical columns
        ('classifier', classifier)
        ])

    # Fit and predict
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    y_prob = pipeline.predict_proba(X_test)[:, 1]

    # Binarize labels
    lb = LabelBinarizer()
    y_test_bin = lb.fit_transform(y_test).ravel()

    print(f"Label encoding: {lb.classes_[0]} -> 0, {lb.classes_[1]} -> 1")

    # Evaluate performance
    auc_score, accuracy, precision, recall, f1, tn, tp, fn, fp, fpr, fnr = model_performance(y_test, y_pred, y_prob)

    # Define model type for labeling columns
    model_type = "RandomForest"

    # Results DataFrame
    test_indices = X_test.index
    results = pd.DataFrame({
        'TrueLabel': y_test.values,
        'TrueLabel_Binary': y_test_bin,
        f'Prediction: {model_type}': y_pred,
        f'Probability: {model_type}': y_prob
    }, index=test_indices)

    return results, auc_score, accuracy, precision, recall, f1, tn, tp, fn, fp, fpr, fnr


# %%
def preprocessing(X_encoded, y_encoded, metadata, train_df):
    preprocessed_data = train_df.copy()
    
    # Imputation
    if mean_imputation_preprocessing_bool == True:
        preprocessed_data = mean_imputation(preprocessed_data)
        print("Mean imputation done.")

    # Features selection
    preprocessed_data = random_forest_feature_selection(X_encoded, 
                                            y_encoded, 
                                            df=preprocessed_data, 
                                            feature_percentage=feature_selection_percentage_to_keep,  # Percentage of features to keep
                                            file=file)

    print(preprocessed_data)
    # Reducing number of features
    if PCA_feature_reduction_bool == True:
        preprocessed_data, pca_model, scaler_model = reduce_dimensionality_pca(preprocessed_data, 
                                                                            to_drop=metadata, 
                                                                            target_column='diagnosis', 
                                                                            percentage=0.3)
        print("Feature reduction with PCA done.")
        
    if removing_low_variance_features_bool == True:
        preprocessed_data = remove_low_variance_columns(preprocessed_data, threshold=0.005)
        print("Low variance features removed.")

    # Standardization and Normalization
    if standardization_preprocessing_bool == True:
        preprocessed_data = standardization(preprocessed_data)
        print("Standardization done.")

    if max_absolute_scaling_preprocessing_bool == True:
        preprocessed_data = max_absolute_scaling(preprocessed_data)
        print("Max absolute scaling done.")
        
    if mean_centering_preprocessing_bool == True:
        preprocessed_data = mean_centering(preprocessed_data)
        print("Mean centering done.")
        
    if min_max_scaling_preprocessing_bool == True:
        preprocessed_data = min_max_scaling(preprocessed_data)
        print("MinMax scaling done.")

    # Transformation
    if log_normal_transformation_transformation_bool == True:
        preprocessed_data = log_normal_transformation(preprocessed_data)
        print("Log normal transformation done.")
        
    if boxcox_transformation_transformation_bool == True:
        preprocessed_data = boxcox_transformation(preprocessed_data) # The Box-Cox transformation can only be applied to strictly positive data
        print("BoxCox transformation done.")
        
    if quantile_transformer_transformation_bool == True:
        preprocessed_data = quantile_transformer(preprocessed_data, n_quantiles=100, output_distribution='uniform')
        print("Quantile transformation done.")

    return preprocessed_data

# %%
"""
### Workflow
"""

# %%
"""
#### Loading and exploring the data
"""

# %%
import pandas as pd

if controls == False:
    data = pd.read_csv(f"datasets/{file}", low_memory=False).set_index("run")
else: data = pd.read_csv(f"controls/{file}", low_memory=False).set_index("run")

metadata = ["locationCountry", "locationContinent", "instrument", "bioProject"]
if controls == False:
    index_names = data[ (data['diagnosis'] != "TD") & (data['diagnosis'] != 'ASD')].index
    data.drop(index_names, inplace = True)
print(data.shape)
data.head()

# %%
try:
    bioprojects = data["bioProject"].value_counts()
    print(bioprojects)
except:pass

# %%
y = data["diagnosis"]

# %%
pie_charts(data)

# %%
"""
#### Cleaning
"""

# %%
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()

cleaned_data = data.copy()
#target = cleaned_data["diagnosis"]

cleaned_data = samples_selection(cleaned_data, samples_threshold) # Percentage of samples to drop
cleaned_data = data_cleaning(cleaned_data, feature_threshold, metadata) # Percentage of features to drop

print("Dataframes shapes:")
print("Raw data: ", data.shape)
print("Cleaned data: ", cleaned_data.shape)

# %%
print("PERMANOVA")
run_permanova(dataset=cleaned_data, feature="diagnosis", to_drop=metadata)
print("\nPLS")
pls_variance_explained(data=cleaned_data, factor_variable="diagnosis", to_drop=metadata)

# %%
try:
    metadata_df = cleaned_data[metadata]
    display(metadata_df.head())
except:pass

# %%
"""
#### Splitting
"""

# %%
from sklearn.model_selection import train_test_split
X = cleaned_data.drop("diagnosis", axis=1)
y = cleaned_data["diagnosis"]
X_train, X_test, y_train, y_test = train_test_split(X, 
                                                    y, 
                                                    test_size=0.2, 
                                                    random_state=42,
                                                    shuffle=True,
                                                    stratify=y) # Ensures same proportion of classes in train and test subset


# %%
"""
#### Prediction
"""

# %%
from sklearn.base import BaseEstimator, TransformerMixin

class CustomPreprocessing(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        metadata,
        feature_selection_pct=0.5,
        pca_feature_reduction=False,
        removing_low_variance_features=False,
        standardization_preprocessing=False,
        max_absolute_scaling_preprocessing=False,
        mean_centering_preprocessing=False,
        min_max_scaling_preprocessing=False,
        log_normal_transformation=False,
        boxcox_transformation=False,
        quantile_transformer_transformation=False,
        mean_imputation_preprocessing=False
    ):
        self.metadata = metadata
        self.feature_selection_pct = feature_selection_pct

        # Store all preprocessing options
        self.pca_feature_reduction = pca_feature_reduction
        self.removing_low_variance_features = removing_low_variance_features
        self.standardization_preprocessing = standardization_preprocessing
        self.max_absolute_scaling_preprocessing = max_absolute_scaling_preprocessing
        self.mean_centering_preprocessing = mean_centering_preprocessing
        self.min_max_scaling_preprocessing = min_max_scaling_preprocessing
        self.log_normal_transformation = log_normal_transformation
        self.boxcox_transformation = boxcox_transformation
        self.quantile_transformer_transformation = quantile_transformer_transformation
        self.mean_imputation_preprocessing = mean_imputation_preprocessing

        # Will be set during fit
        self.feature_names = None
        self.y_ = None

    def fit(self, X, y=None):
        import pandas as pd

        self.y_ = y
        if isinstance(X, pd.DataFrame):
            self.feature_names = X.columns

        # Prepare full preprocessed training data to store final column names
        X_encoded = X.drop(columns=self.metadata, errors="ignore")
        df = X_encoded.copy()
        if y is not None:
            df["diagnosis"] = pd.Series(y, index=X.index, name="diagnosis")

        processed = self._preprocessing_pipeline(X_encoded, y, self.metadata, df)
        self.final_columns_ = processed.drop(columns=["diagnosis"], errors="ignore").columns.tolist()

        return self

    def transform(self, X):
        import pandas as pd
        import numpy as np
    
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X, columns=self.feature_names)
    
        X = X.drop(columns=self.metadata, errors="ignore")
    
        cat_cols = X.select_dtypes(include=["object", "category"]).columns
        if len(cat_cols) > 0:
            X = pd.get_dummies(X, columns=cat_cols, drop_first=True)
    
        if self.y_ is not None and len(X) == len(self.y_):
            df = X.copy()
            df["diagnosis"] = pd.Series(self.y_, index=X.index, name="diagnosis")
    
            processed = self._preprocessing_pipeline(X, self.y_, metadata=self.metadata, train_df=df)
            processed = processed.drop(columns=["diagnosis"], errors="ignore")
        else:
            processed = X
    
        # Align columns to training set
        processed = processed.reindex(columns=self.final_columns_, fill_value=0)
    
        return processed

    def _preprocessing_pipeline(self, X_encoded, y_encoded, metadata, train_df):
        # Assume all helper functions below are defined elsewhere and imported
        df = train_df.copy()
        # Drop metadata columns
        df = df.drop(columns=metadata, errors="ignore")
        # --- Encode categorical features ---
        cat_cols = df.select_dtypes(include=["object", "category"]).columns
        df = pd.get_dummies(df, columns=cat_cols, drop_first=True)

        df = random_forest_feature_selection(
            X_encoded,
            y_encoded,
            df=df,
            feature_percentage=self.feature_selection_pct,
            file=None
        )

        if self.pca_feature_reduction:
            df, self.pca_model, self.scaler_model = reduce_dimensionality_pca(
                df,
                to_drop=metadata,
                target_column="diagnosis",
                percentage=0.1
            )

        if self.removing_low_variance_features:
            df = remove_low_variance_columns(df, threshold=low_variance_threshold)

        if self.standardization_preprocessing:
            df = standardization(df)

        if self.max_absolute_scaling_preprocessing:
            df = max_absolute_scaling(df)

        if self.mean_centering_preprocessing:
            df = mean_centering(df)

        if self.min_max_scaling_preprocessing:
            df = min_max_scaling(df)

        if self.log_normal_transformation:
            df = log_normal_transformation(df)

        if self.boxcox_transformation:
            df = boxcox_transformation(df)

        if self.quantile_transformer_transformation:
            df = quantile_transformer(df, n_quantiles=100, output_distribution='uniform')

        if self.mean_imputation_preprocessing:
            df = mean_imputation(df)

        display(df.head())
        return df


# %%
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators=1000, random_state=42, min_samples_split=10)

custom_preprocessor = CustomPreprocessing(
    metadata=metadata,
    feature_selection_pct=feature_selection_percentage_to_keep,
    pca_feature_reduction=PCA_feature_reduction_bool,
    removing_low_variance_features=removing_low_variance_features_bool,
    standardization_preprocessing=standardization_preprocessing_bool,
    min_max_scaling_preprocessing=min_max_scaling_preprocessing_bool,
    max_absolute_scaling_preprocessing=max_absolute_scaling_preprocessing_bool,
    mean_centering_preprocessing=mean_centering_preprocessing_bool,
    log_normal_transformation=log_normal_transformation_transformation_bool,
    boxcox_transformation=boxcox_transformation_transformation_bool,
    quantile_transformer_transformation=quantile_transformer_transformation_bool,
    mean_imputation_preprocessing=mean_imputation_preprocessing_bool
)

pipeline = Pipeline([
    ("custom", custom_preprocessor),
    ("clf", classifier)
])

# %%
print("PERMANOVA")
run_permanova(dataset=merge_dataframes(y_train, X_train), feature="diagnosis", to_drop=metadata)
print("PLS")
pls_variance_explained(data=merge_dataframes(y_train, X_train), factor_variable="diagnosis", to_drop=metadata)

# %%
pipeline.fit(X_train, y_train)

# %%
y_pred = pipeline.predict(X_test)
y_prob = pipeline.predict_proba(X_test)[:, 0] # TD:0, ASD:1

# %%
auc_score, accuracy, precision, recall, f1, tn, tp, fn, fp, fpr, fnr = model_performance(y_test, y_pred, y_prob)

# %%
"""
### Predictions and Final Results
"""

# %%
# Creating dataframe
import pandas as pd
#parameters_df = pd.DataFrame()
parameters_df = pd.read_excel("parameters-performance.xlsx")
try: parameters_df = parameters_df.drop("Unnamed: 0", axis=1)
except: pass
parameters = {
    "file": [file],
    "model": [f"{model_type}, {classifier}"],
    "feature_selection_percentage_to_keep": [feature_selection_percentage_to_keep],
    "samples_threshold": [samples_threshold],
    "feature_threshold": [feature_threshold],
    "PCA_feature_reduction": [PCA_feature_reduction_bool],
    "Removing_low_variance_features": [f"{removing_low_variance_features_bool}, threshold: {low_variance_threshold}"],
    "standardization_preprocessing": [standardization_preprocessing_bool],
    "min_max_scaling_prepricessing": [min_max_scaling_preprocessing_bool],
    "max_absolute_scaling_prepricessing": [max_absolute_scaling_preprocessing_bool],
    "mean_centering_prepricessing": [mean_centering_preprocessing_bool],
    "log_normal_transformation_transformation": [log_normal_transformation_transformation_bool],
    "boxcox_transformation_transformation": [boxcox_transformation_transformation_bool],
    "quantile_transformer_transformation": [quantile_transformer_transformation_bool],
    "mean_imputation_prepricessing": [mean_imputation_preprocessing_bool],
    "auc_score": auc_score, 
    "accuracy": accuracy, 
    "precision": precision, 
    "recall": recall, 
    "f1": f1, 
    "tn": tn, 
    "tp": tp, 
    "fn": fn, 
    "fp": fp, 
    "fpr": fpr, 
    "fnr": fnr
}

#parameters_df = pd.DataFrame(parameters)

new_row = pd.DataFrame(parameters)
parameters_df = pd.concat([parameters_df, new_row], ignore_index=True)

parameters_df.to_excel("parameters-performance.xlsx")
display(parameters_df)