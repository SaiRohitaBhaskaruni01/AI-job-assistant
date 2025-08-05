#!/usr/bin/env python3
"""
Minimal Data Preprocessor - Keep only essential columns for RAG job search
Clean, focused dataset with just what we need
"""
import pandas as pd
import os


def preprocess_minimal_job_data(input_path="data/clean_jobs.csv", output_path="data/jobs_minimal.csv"):
    """
    Create a minimal, clean dataset with only essential columns
    """
    print("🎯 MINIMAL JOB DATA PREPROCESSING")
    print("=" * 50)
    
    # Step 1: Load data
    print("📂 Loading data...")
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input file not found: {input_path}")
    
    df = pd.read_csv(input_path)
    original_count = len(df)
    print(f"✅ Loaded {original_count:,} records")
    print(f"📊 Original columns: {list(df.columns)}")
    
    # Step 2: Select ONLY essential columns
    essential_columns = [
        'id',           # ✅ Unique identifier  
        'title',        # ✅ Job title (main filtering field)
        'company',      # ✅ Company name (display & context)
        'location',     # ✅ Job location (display & filtering)
        'description'   # ✅ Job description (RAG content)
    ]
    
    print(f"\n🎯 Keeping ONLY {len(essential_columns)} essential columns:")
    for i, col in enumerate(essential_columns, 1):
        purpose = {
            'id': 'Unique job identifier',
            'title': 'Job title - main filtering field', 
            'company': 'Company name - display & context',
            'location': 'Job location - display & filtering',
            'description': 'Job description - RAG content'
        }
        print(f"  {i}. {col} - {purpose[col]}")
    
    # Check if all essential columns exist
    missing_cols = [col for col in essential_columns if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing essential columns: {missing_cols}")
    
    # Keep only essential columns
    df = df[essential_columns].copy()
    
    dropped_cols = [col for col in ['link', 'source', 'date_posted', 'work_type', 'employment_type'] 
                   if col in df.columns]
    print(f"\n🗑️ Dropped unnecessary columns: {dropped_cols}")
    
    # Step 3: Clean data
    print(f"\n🧹 Cleaning data...")
    
    # Remove rows with nulls in ANY essential field
    before_count = len(df)
    df = df.dropna()  # Drop any row with any null value
    after_count = len(df)
    removed = before_count - after_count
    
    print(f"✅ Removed {removed:,} rows with missing data")
    print(f"📊 Remaining: {after_count:,} complete records")
    
    # Remove exact duplicates
    before_count = len(df)
    df = df.drop_duplicates(keep='first')
    after_count = len(df)
    removed = before_count - after_count
    
    print(f"✅ Removed {removed:,} exact duplicates")
    print(f"📊 Remaining: {after_count:,} unique records")
    
    # Step 4: Clean text fields
    print(f"\n🧽 Cleaning text fields...")
    
    # Clean each text field
    for col in ['title', 'company', 'location', 'description']:
        # Strip whitespace and normalize spaces
        df[col] = df[col].astype(str).str.strip()
        df[col] = df[col].str.replace(r'\s+', ' ', regex=True)
    
    # Remove jobs with very short descriptions (likely incomplete)
    before_count = len(df)
    min_desc_length = 50  # At least 50 characters
    df = df[df['description'].str.len() >= min_desc_length]
    after_count = len(df)
    removed = before_count - after_count
    
    print(f"✅ Removed {removed:,} jobs with short descriptions (< {min_desc_length} chars)")
    
    # Standardize common location patterns
    df.loc[df['location'].str.contains('remote|work from home|wfh', case=False, na=False), 'location'] = 'Remote'
    
    print("✅ Standardized remote locations")
    
    # Step 5: Add one derived field for better title matching
    df['title_normalized'] = df['title'].str.lower().str.strip()
    print("✅ Added title_normalized for better matching")
    
    # Step 6: Final quality check
    print(f"\n🔍 Final data quality check...")
    final_count = len(df)
    
    # Verify no nulls remain
    null_check = df.isnull().sum()
    if null_check.any():
        print(f"⚠️ Warning: Still have nulls: {null_check[null_check > 0]}")
    else:
        print("✅ No null values in final dataset")
    
    # Show data distribution
    print(f"\n📊 Final dataset statistics:")
    print(f"  Total jobs: {final_count:,}")
    print(f"  Unique companies: {df['company'].nunique():,}")
    print(f"  Unique locations: {df['location'].nunique():,}")
    print(f"  Unique job titles: {df['title'].nunique():,}")
    print(f"  Avg description length: {df['description'].str.len().mean():.0f} chars")
    
    # Step 7: Save minimal dataset
    print(f"\n💾 Saving minimal dataset...")
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Reset index and save
    df = df.reset_index(drop=True)
    df.to_csv(output_path, index=False)
    
    # Step 8: Show summary
    print("\n" + "=" * 50)
    print("📊 MINIMAL PREPROCESSING SUMMARY")
    print("=" * 50)
    print(f"Original records: {original_count:,}")
    print(f"Final records: {final_count:,}")
    print(f"Data reduction: {((original_count - final_count) / original_count) * 100:.1f}%")
    print(f"Final columns: {len(df.columns)} (was {len(essential_columns) + len(dropped_cols) if dropped_cols else len(essential_columns)})")
    print(f"Output file: {output_path}")
    print(f"File size: {os.path.getsize(output_path) / 1024 / 1024:.1f} MB")
    
    # Show sample data  
    print(f"\n📋 Sample of minimal dataset:")
    sample_df = df[['id', 'title', 'company', 'location']].head(3)
    print(sample_df.to_string(index=False))
    
    # Show top categories for verification
    print(f"\n🎯 Top 5 Job Titles:")
    for title, count in df['title'].value_counts().head(5).items():
        print(f"  • {title}: {count} jobs")
    
    print(f"\n🏢 Top 5 Companies:")
    for company, count in df['company'].value_counts().head(5).items():
        print(f"  • {company}: {count} jobs")
    
    print(f"\n📍 Top 5 Locations:")
    for location, count in df['location'].value_counts().head(5).items():
        print(f"  • {location}: {count} jobs")
    
    print(f"\n🎉 Minimal preprocessing completed!")
    print(f"✅ Clean, focused dataset ready for RAG: {output_path}")
    
    return df


def analyze_minimal_output(file_path="data/jobs_minimal.csv"):
    """
    Quick analysis of the minimal dataset
    """
    print("\n" + "=" * 50)
    print("🔍 ANALYZING MINIMAL DATASET")
    print("=" * 50)
    
    if not os.path.exists(file_path):
        print(f"❌ File not found: {file_path}")
        return
    
    df = pd.read_csv(file_path)
    
    print(f"📊 Dataset: {len(df):,} records, {len(df.columns)} columns")
    print(f"📋 Columns: {list(df.columns)}")
    print(f"💾 File size: {os.path.getsize(file_path) / 1024:.1f} KB")
    
    # Check data completeness
    print(f"\n✅ Data Completeness:")
    for col in df.columns:
        complete_pct = ((len(df) - df[col].isnull().sum()) / len(df)) * 100
        print(f"  {col}: {complete_pct:.1f}% complete")
    
    # Show memory usage
    print(f"\n💾 Memory Usage:")
    memory_usage = df.memory_usage(deep=True)
    for col, usage in memory_usage.items():
        if col != 'Index':
            print(f"  {col}: {usage / 1024:.1f} KB")
    
    print(f"\n🚀 Ready for RAG processing!")


if __name__ == "__main__":
    try:
        # Run minimal preprocessing
        df = preprocess_minimal_job_data()
        
        # Analyze the output
        analyze_minimal_output()
        
        print(f"\n" + "=" * 60)
        print("🎯 WHAT'S INCLUDED IN YOUR MINIMAL DATASET:")
        print("=" * 60)
        print("✅ id - Job identifier")
        print("✅ title - Job title (for filtering)")
        print("✅ company - Company name (for display)")
        print("✅ location - Job location (for display/filtering)")
        print("✅ description - Full job description (for RAG)")
        print("✅ title_normalized - Lowercase title (for matching)")
        print("")
        print("🗑️ REMOVED:")
        print("❌ link - Not needed for search")
        print("❌ source - Not essential")
        print("❌ date_posted - Not essential for matching")
        print("❌ work_type - Was 100% null anyway")
        print("❌ employment_type - Was 100% null anyway")
        print("")
        print("🚀 Perfect for RAG-powered job search!")
        
    except Exception as e:
        print(f"❌ Preprocessing failed: {e}")
        import traceback
        traceback.print_exc()