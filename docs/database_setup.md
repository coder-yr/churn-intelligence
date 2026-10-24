# Supabase Cloud PostgreSQL Integration & Setup Guide

**Project**: Customer Lifetime Value and Churn Intelligence System  
**Database**: Managed Cloud PostgreSQL (Supabase)  
**ORM / Driver**: SQLAlchemy 2.0 with Psycopg 3 (`postgresql+psycopg://`)

---

## 1. Overview & Architecture

To support real-world, cloud-native deployments without the operational burden of local server maintenance, this system uses **Supabase Cloud PostgreSQL** as its centralized data store.

### Architectural Flow:
1. **Source Data**: Cleaned dataset in `data/processed/telco_churn_cleaned.csv` (7,043 rows, 22 columns).
2. **Ingestion Engine**: [`src/database/load_data.py`](file:///f:/yash/projects/nexvia/churn-intelligence/src/database/load_data.py) validates data integrity, manages transactions, and executes batch upserts (`ON CONFLICT (customerID) DO UPDATE`).
3. **Storage**: Supabase `customers` table with indexed primary key (`customerID`), indexed `churn_label`, and numeric check constraints.
4. **Application Consumer**: The Streamlit dashboard and feature engineering pipelines query Supabase using SQLAlchemy session management.

---

## 2. Supabase Cloud Configuration

### Step 1: Create a Free Project
1. Log in to [supabase.com](https://supabase.com).
2. Click **"New Project"**.
3. Choose a project name (e.g. `churn-intelligence`).
4. Set a strong database password and store it in your password manager.
5. Choose a region geographically close to you (e.g., `South Asia (Mumbai)`).
6. Click **"Create new project"**.

### Step 2: Retrieve the Connection String
1. In your project dashboard, click the green **"Connect"** button in the top navigation (or go to **Project Settings** ⚙️ ➔ **Database** ➔ **Connection Pooling**).
2. Select the **"URI"** tab.
3. Set the mode to **"Session"** (Port `5432`).
4. Copy the connection URI:
   ```text
   postgresql://postgres.[PROJECT-REF]:[YOUR-PASSWORD]@aws-0-[REGION].pooler.supabase.com:5432/postgres
   ```
5. Change the URI prefix from `postgresql://` to `postgresql+psycopg://` so SQLAlchemy uses the modern Psycopg 3 driver.

> [!IMPORTANT]
> **Password URL Encoding**:
> If your database password contains characters like `@`, `#`, `:`, `/`, or `%`, you must URL-encode them:
> - `#` ➔ `%23`
> - `@` ➔ `%40`
> - `:` ➔ `%3A`
> - `/` ➔ `%2F`

---

## 3. Local Environment Setup

### Step 1: Configure `.env`
Ensure your local `.env` file (which is git-ignored) contains your real connection string:
```bash
# In .env (do NOT commit this file to Git):
DATABASE_URL=postgresql+psycopg://postgres.[PROJECT-REF]:[YOUR-PASSWORD]@aws-0-[REGION].pooler.supabase.com:5432/postgres
```

### Step 2: Test the Connection
Run the standalone connection test:
```powershell
.\.venv\Scripts\python.exe -m src.database.connection
```
**Expected output upon success**:
```text
============================================================
SUPABASE POSTGRESQL CONNECTION TEST
============================================================
 - connected: True
 - database: postgres
 - user: postgres.[PROJECT-REF]
 - postgres_version: PostgreSQL 15...
 - message: Successfully connected to PostgreSQL database.
============================================================
```

---

## 4. Ingesting Cleaned Data into Supabase

Run the idempotent batch ingestion pipeline:
```powershell
.\.venv\Scripts\python.exe -m src.database.load_data
```

### Ingestion Safeguards:
- **Pre-check Validation**: Confirms 0 nulls, correct data types, valid churn labels (`0` or `1`), and 100% unique customer IDs before touching the database.
- **Auto Table Creation**: Creates the `customers` table with constraints if it does not exist.
- **Idempotent Upsert**: Uses PostgreSQL `ON CONFLICT (customerID) DO UPDATE` so you can safely rerun the script without duplicate rows or primary key collision errors.
- **Batch Processing**: Streams records in 1,000-row chunks for fast and stable network transmission.
- **Automated Verification**: Runs SQL `COUNT` queries post-ingestion to verify:
  - Total records: **7,043**
  - Unique IDs: **7,043**
  - Churned count: **1,869**
  - Retained count: **5,174**

---

## 5. Running Automated Tests

Run the unit test suite without needing an active internet connection or live credentials:
```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_database_validation.py -v
```
All 13 unit tests validate URL masking, placeholder rejection, CSV data integrity, and SQLAlchemy in-memory table compilation.

---

## 6. Troubleshooting Common Issues

| Error | Root Cause | Solution |
| :--- | :--- | :--- |
| `DATABASE_URL contains unconfigured template placeholders` | `.env` still has default `[YOUR-PASSWORD]` placeholder text. | Open `.env` and replace placeholders with your actual Supabase credentials. |
| `connection to server at ... failed: FATAL: password authentication failed` | Incorrect password or unescaped special characters. | URL-encode special characters in your password (e.g. `%23` for `#`). |
| `Tenant or user not found` | Missing project-ref in user string when connecting via pooler. | Ensure the username is `postgres.[PROJECT-REF]` when connecting through `aws-0-*.pooler.supabase.com`. |
| `Could not translate host name` | DNS resolution failure or offline network. | Check your internet connection and verify the pooler URL in your Supabase dashboard. |
| `sslmode=require error` | Remote connection requires TLS/SSL encryption. | Append `?sslmode=require` to your `DATABASE_URL`. |
