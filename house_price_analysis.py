# =========== PART 1: COMPREHENSIVE EDA ===========
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Load and basic info - FIXED WITH YOUR FILE PATH
df = pd.read_csv('../data/regression_data.xls')
print(f"Dataset: {df.shape[0]} rows, {df.shape[1]} columns")

# Check what columns we have
print(f"\nColumns in dataset:")
print(df.columns.tolist())

# 2. Check missing values
print("\n" + "="*60)
print("MISSING VALUES ANALYSIS")
print("="*60)
missing_counts = df.isnull().sum()
missing_percentage = (missing_counts / len(df)) * 100
missing_df = pd.DataFrame({
    'Missing Count': missing_counts,
    'Missing %': missing_percentage
}).sort_values('Missing Count', ascending=False)

print(missing_df[missing_df['Missing Count'] > 0])

if missing_counts.sum() == 0:
    print("✓ No missing values found!")

# 3. Statistical summary
print("\n" + "="*60)
print("BASIC STATISTICAL SUMMARY")
print("="*60)

# Identify numerical and categorical columns
numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
categorical_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()

print(f"Numerical columns ({len(numerical_cols)}): {numerical_cols}")
print(f"Categorical columns ({len(categorical_cols)}): {categorical_cols}")

# Basic stats for key numerical variables (including price if it exists)
if 'price' in df.columns:
    key_vars = ['price', 'sqft_living', 'grade', 'bathrooms', 'bedrooms', 
                'sqft_lot', 'condition', 'waterfront', 'sqft_above', 
                'sqft_living15', 'sqft_lot15']
else:
    # Find similar column names
    price_col = None
    for col in df.columns:
        if 'price' in col.lower():
            price_col = col
            break
    
    if price_col:
        key_vars = [price_col, 'sqft_living', 'grade', 'bathrooms', 'bedrooms']
    else:
        # Use first 10 numerical columns
        key_vars = numerical_cols[:10]

print(f"\nBasic statistics for key variables:")
print(df[key_vars].describe().round(2))

# 4. Target variable analysis (assuming 'price' is the target)
print("\n" + "="*60)
print("TARGET VARIABLE ANALYSIS (PRICE)")
print("="*60)

# Find price column
price_col = 'price' if 'price' in df.columns else None
if not price_col:
    for col in df.columns:
        if 'price' in col.lower():
            price_col = col
            break

if price_col:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # Original distribution
    axes[0].hist(df[price_col], bins=50, edgecolor='black', alpha=0.7)
    axes[0].set_title(f'{price_col} Distribution')
    axes[0].set_xlabel(price_col)
    axes[0].set_ylabel('Frequency')
    axes[0].axvline(df[price_col].mean(), color='red', linestyle='--', label=f'Mean: ${df[price_col].mean():,.0f}')
    axes[0].axvline(df[price_col].median(), color='green', linestyle='--', label=f'Median: ${df[price_col].median():,.0f}')
    axes[0].legend()
    
    # Boxplot
    axes[1].boxplot(df[price_col])
    axes[1].set_title(f'{price_col} Boxplot')
    axes[1].set_ylabel(price_col)
    
    # Log transformation (if no negative/zero prices)
    if (df[price_col] > 0).all():
        axes[2].hist(np.log1p(df[price_col]), bins=50, edgecolor='black', alpha=0.7)
        axes[2].set_title(f'Log-Transformed {price_col}')
        axes[2].set_xlabel(f'log({price_col})')
        axes[2].set_ylabel('Frequency')
    else:
        # If negative values exist, show cumulative distribution
        axes[2].hist(df[price_col], bins=50, cumulative=True, edgecolor='black', alpha=0.7)
        axes[2].set_title(f'Cumulative {price_col} Distribution')
        axes[2].set_xlabel(price_col)
        axes[2].set_ylabel('Cumulative Frequency')
    
    plt.tight_layout()
    plt.show()
    
    # Price statistics
    print(f"\n{price_col} Statistics:")
    print(f"  Mean: ${df[price_col].mean():,.2f}")
    print(f"  Median: ${df[price_col].median():,.2f}")
    print(f"  Std Dev: ${df[price_col].std():,.2f}")
    print(f"  Min: ${df[price_col].min():,.2f}")
    print(f"  Max: ${df[price_col].max():,.2f}")
    print(f"  Skewness: {df[price_col].skew():.2f}")
    print(f"  Kurtosis: {df[price_col].kurtosis():.2f}")
else:
    print("Warning: Could not find price column in dataset")
    print(f"Available columns: {df.columns.tolist()}")

# 5. Correlation analysis
print("\n" + "="*60)
print("CORRELATION ANALYSIS")
print("="*60)

if price_col and len(numerical_cols) > 1:
    # Calculate correlation
    corr_matrix = df[numerical_cols].corr()
    
    # Get correlation with price
    if price_col in corr_matrix.columns:
        price_corr = corr_matrix[price_col].sort_values(ascending=False)
        print("\nTop 10 correlations with price:")
        print(price_corr.head(10))
        
        # Visualize correlation heatmap
        plt.figure(figsize=(12, 8))
        top_corr_features = price_corr.index[:10].tolist()
        top_corr_matrix = df[top_corr_features].corr()
        
        sns.heatmap(top_corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', 
                    center=0, square=True, linewidths=1, cbar_kws={"shrink": 0.8})
        plt.title('Top 10 Features Correlation with Price', fontsize=14)
        plt.tight_layout()
        plt.show()
    else:
        print(f"Price column '{price_col}' not found in numerical columns")
else:
    print("Not enough numerical columns for correlation analysis")

# 6. High-value properties analysis (≥ $650K)
print("\n" + "="*60)
print("HIGH-VALUE PROPERTIES ANALYSIS (≥ $650K)")
print("="*60)

if price_col:
    df['high_value'] = df[price_col] >= 650000
    high_value = df[df['high_value']]
    regular_value = df[~df['high_value']]
    
    print(f"High-value properties (≥ $650K): {len(high_value):,}")
    print(f"Regular properties (< $650K): {len(regular_value):,}")
    print(f"Percentage high-value: {len(high_value)/len(df)*100:.1f}%")
    
    print("\nComparison - High Value vs Regular Properties:")
    
    # Define comparison columns based on available data
    possible_comp_cols = ['grade', 'sqft_living', 'bathrooms', 'bedrooms', 
                         'waterfront', 'view', 'condition', 'sqft_above',
                         'sqft_lot', 'floors', 'yr_built']
    
    comp_cols = [col for col in possible_comp_cols if col in df.columns]
    
    if comp_cols:
        comparison_data = []
        for col in comp_cols:
            hv_mean = high_value[col].mean() if len(high_value) > 0 else 0
            rv_mean = regular_value[col].mean() if len(regular_value) > 0 else 0
            
            if rv_mean != 0:  # Avoid division by zero
                pct_diff = ((hv_mean - rv_mean) / rv_mean) * 100
            else:
                pct_diff = 0
                
            comparison_data.append({
                'Feature': col,
                'High_Value_Mean': hv_mean,
                'Regular_Mean': rv_mean,
                'Difference': hv_mean - rv_mean,
                'Pct_Difference (%)': pct_diff
            })
        
        comparison_df = pd.DataFrame(comparison_data)
        print(comparison_df.to_string(index=False))
        
        # Visualize top differences
        top_diff = comparison_df.nlargest(5, 'Pct_Difference (%)')
        if len(top_diff) > 0:
            fig, ax = plt.subplots(figsize=(10, 6))
            y_pos = np.arange(len(top_diff))
            ax.barh(y_pos, top_diff['Pct_Difference (%)'])
            ax.set_yticks(y_pos)
            ax.set_yticklabels(top_diff['Feature'])
            ax.set_xlabel('Percentage Difference (%)')
            ax.set_title('Top Features Differentiating High-Value Properties')
            ax.invert_yaxis()  # Highest on top
            plt.tight_layout()
            plt.show()
    else:
        print("No comparison columns found in dataset")
else:
    print(f"Cannot analyze high-value properties: Price column not found")

# =========== PART 2: MODELING (Based on EDA insights) ===========
print("\n" + "="*60)
print("MODELING PHASE")
print("="*60)

if price_col and len(numerical_cols) > 2:
    from sklearn.model_selection import train_test_split, cross_val_score
    from sklearn.preprocessing import StandardScaler
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.linear_model import LinearRegression, Ridge
    from sklearn.tree import DecisionTreeRegressor
    from sklearn.neighbors import KNeighborsRegressor
    from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
    
    # Select features based on correlation analysis
    if 'price_corr' in locals() and len(price_corr) > 1:
        # Use top 10 features excluding price itself
        top_features = price_corr.index[1:11].tolist() if len(price_corr) > 10 else price_corr.index[1:].tolist()
    else:
        # Fallback: use all numerical columns except price and ID
        top_features = [col for col in numerical_cols if col != price_col and 'id' not in col.lower()]
        top_features = top_features[:10]  # Take first 10
    
    print(f"\nSelected features for modeling ({len(top_features)}):")
    print(top_features)
    
    X = df[top_features]
    y = df[price_col]
    
    # Handle any missing values
    if X.isnull().sum().sum() > 0:
        print(f"\nFilling {X.isnull().sum().sum()} missing values with median...")
        X = X.fillna(X.median())
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    print(f"\nTrain set: {X_train.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train multiple models
    models = {
        'Linear Regression': LinearRegression(),
        'Ridge Regression': Ridge(alpha=1.0),
        'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
        'Decision Tree': DecisionTreeRegressor(random_state=42),
        'K-Nearest Neighbors': KNeighborsRegressor(n_neighbors=5)
    }
    
    print("\n" + "="*60)
    print("MODEL PERFORMANCE COMPARISON")
    print("="*60)
    
    results = []
    for name, model in models.items():
        print(f"\nTraining {name}...")
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        # Cross-validation
        cv_scores = cross_val_score(model, X_train_scaled, y_train, 
                                    cv=5, scoring='r2', n_jobs=-1)
        
        results.append({
            'Model': name,
            'RMSE': f"${rmse:,.0f}",
            'MAE': f"${mae:,.0f}",
            'R²': f"{r2:.3f}",
            'CV R² Mean': f"{cv_scores.mean():.3f}",
            'CV R² Std': f"{cv_scores.std():.3f}"
        })
        
        print(f"  R²: {r2:.3f}")
        print(f"  RMSE: ${rmse:,.0f}")
        print(f"  MAE: ${mae:,.0f}")
        print(f"  CV R²: {cv_scores.mean():.3f} (±{cv_scores.std():.3f})")
    
    # Display results
    results_df = pd.DataFrame(results)
    print("\n" + "="*60)
    print("SUMMARY TABLE")
    print("="*60)
    print(results_df.to_string(index=False))
    
    # Feature importance from Random Forest
    print("\n" + "="*60)
    print("FEATURE IMPORTANCE (Random Forest)")
    print("="*60)
    
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_model.fit(X_train_scaled, y_train)
    
    feature_importance = pd.DataFrame({
        'Feature': top_features,
        'Importance': rf_model.feature_importances_
    }).sort_values('Importance', ascending=False)
    
    print(feature_importance.to_string(index=False))
    
    # Visualize feature importance
    plt.figure(figsize=(10, 6))
    plt.barh(feature_importance['Feature'], feature_importance['Importance'])
    plt.xlabel('Importance')
    plt.title('Feature Importance - Random Forest')
    plt.gca().invert_yaxis()  # Highest importance on top
    plt.tight_layout()
    plt.show()
    
    # =========== PART 3: BUSINESS INSIGHTS ===========
    print("\n" + "="*60)
    print("KEY BUSINESS INSIGHTS FOR MANAGEMENT")
    print("="*60)
    
    if 'high_value' in df.columns:
        print("\n1. HIGH-VALUE PROPERTIES (≥ $650K):")
        print(f"   • Represent {len(high_value)/len(df)*100:.1f}% of all properties")
        print(f"   • Average price: ${high_value[price_col].mean():,.0f}")
        print(f"   • Key differentiators:")
        
        # Show top 3 differentiating features
        if 'comparison_df' in locals():
            top_diff_features = comparison_df.nlargest(3, 'Pct_Difference (%)')
            for _, row in top_diff_features.iterrows():
                print(f"     - {row['Feature']}: {row['High_Value_Mean']:.1f} vs {row['Regular_Mean']:.1f} ({row['Pct_Difference (%)']:+.1f}%)")
    
    print("\n2. PREDICTIVE MODELING RESULTS:")
    best_model_row = results_df.loc[results_df['R²'].astype(float).idxmax()]
    print(f"   • Best performing model: {best_model_row['Model']}")
    print(f"   • R² Score: {best_model_row['R²']}")
    print(f"   • Average prediction error: {best_model_row['MAE']}")
    
    print("\n3. TOP PRICE PREDICTORS:")
    for i, row in feature_importance.head(3).iterrows():
        print(f"   • {row['Feature']} (importance: {row['Importance']:.3f})")
    
    print("\n4. RECOMMENDATIONS:")
    print("   • Focus on improving top predictors (see above) for property value appreciation")
    print("   • High-value properties show distinct characteristics - target these features")
    print("   • Use the Random Forest model for accurate price predictions")
    print("   • Monitor waterfront and grade as key value drivers")
    
    # Prepare for Tableau (optional)
    print("\n" + "="*60)
    print("TABLEAU PREPARATION")
    print("="*60)
    
    tableau_df = df.copy()
    
    # Create useful derived columns for Tableau
    if price_col:
        tableau_df['price_category'] = np.where(tableau_df[price_col] >= 650000, 
                                               'High (≥$650K)', 'Regular (<$650K)')
        
        # Create price per sqft if sqft_living exists
        if 'sqft_living' in tableau_df.columns:
            tableau_df['price_per_sqft'] = tableau_df[price_col] / tableau_df['sqft_living']
            print("   • Created 'price_per_sqft' column")
        
        print("   • Created 'price_category' column")
    
    # Save for Tableau
    tableau_df.to_csv('../data/house_data_tableau_ready.csv', index=False)
    print(f"   • Tableau-ready data saved to: ../data/house_data_tableau_ready.csv")
    
else:
    print("Skipping modeling phase - insufficient data or price column not found")
    print(f"Price column found: {price_col}")
    print(f"Numerical columns: {len(numerical_cols)}")

print("\n" + "="*60)
print("ANALYSIS COMPLETE!")
print("="*60)