from pathlib import Path

import pandas as pd
import streamlit as st

BASE_DIR = Path(__file__).resolve().parent

st.title("Real Estate Buyer Segmentation Dashboard")
st.caption("Machine Learning Based Buyer Segmentation and Investment Profiling for Real Estate Market Intelligence")

# Prefer the exported, buyer-level dataset that includes buyer_segment.
segmented_paths = [
    BASE_DIR / "clients_segmented.csv",
    BASE_DIR / "data" / "clients_segmented.csv",
]
raw_paths = [BASE_DIR / "data" / "clients.csv", BASE_DIR / "clients.csv"]
source = next((p for p in segmented_paths if p.is_file()), None)
if source is None:
    source = next((p for p in raw_paths if p.is_file()), None)
if source is None:
    st.error("Buyer dataset not found. Put clients_segmented.csv beside app.py, or put clients.csv in the data folder.")
    st.stop()

df = pd.read_csv(source)

# If only the raw dataset exists, merge real buyer-level segment assignments.
if "buyer_segment" not in df.columns:
    assignment_paths = [
        BASE_DIR / "buyer_segments.csv",
        BASE_DIR / "dashboard_data" / "buyer_segments.csv",
        BASE_DIR / "data" / "buyer_segments.csv",
    ]
    assignment_path = next((p for p in assignment_paths if p.is_file()), None)
    if assignment_path is not None and "client_id" in df.columns:
        assignments = pd.read_csv(assignment_path)
        if {"client_id", "buyer_segment"}.issubset(assignments.columns):
            assignments = assignments[["client_id", "buyer_segment"]].drop_duplicates("client_id")
            df = df.merge(assignments, on="client_id", how="left", validate="many_to_one")

required = ["client_id", "first_name", "last_name", "country", "region",
            "acquisition_purpose", "client_type", "satisfaction_score",
            "loan_applied", "referral_channel", "buyer_segment"]
missing = [name for name in required if name not in df.columns]
if missing:
    st.error("Missing dataset columns: " + ", ".join(missing))
    st.info("Export the notebook DataFrame with the buyer_segment column as clients_segmented.csv, then place it beside app.py.")
    st.stop()

if df["buyer_segment"].isna().any():
    st.warning("Some buyer records have no segment assignment; segment charts exclude those records.")

df["satisfaction_score"] = pd.to_numeric(df["satisfaction_score"], errors="coerce")

st.sidebar.header("Filter Buyers")
selected_country = st.sidebar.multiselect("Select Country", sorted(df["country"].dropna().unique().tolist()), key="country_filter")
selected_region = st.sidebar.multiselect("Select Region", sorted(df["region"].dropna().unique().tolist()), key="region_filter")
selected_purpose = st.sidebar.multiselect("Select Acquisition Purpose", sorted(df["acquisition_purpose"].dropna().unique().tolist()), key="purpose_filter")
selected_client_type = st.sidebar.multiselect("Select Client Type", sorted(df["client_type"].dropna().unique().tolist()), key="client_type_filter")

filtered_df = df.copy()
for column, selections in [
    ("country", selected_country),
    ("region", selected_region),
    ("acquisition_purpose", selected_purpose),
    ("client_type", selected_client_type),
]:
    if selections:
        filtered_df = filtered_df[filtered_df[column].isin(selections)]

st.sidebar.write("Matching buyers: {:,}".format(len(filtered_df)))

st.header("Buyer Overview")
count = len(filtered_df)
mean_satisfaction = filtered_df["satisfaction_score"].mean() if count else 0
investment_pct = filtered_df["acquisition_purpose"].astype(str).str.strip().str.lower().eq("investment").mean() * 100 if count else 0
loan_pct = filtered_df["loan_applied"].astype(str).str.strip().str.lower().eq("yes").mean() * 100 if count else 0
c1, c2, c3, c4 = st.columns(4)
c1.metric("Total Buyers", "{:,}".format(count))
c2.metric("Average Satisfaction", "{:.2f}".format(mean_satisfaction))
c3.metric("Investment Buyers", "{:.2f}%".format(investment_pct))
c4.metric("Loan Application Rate", "{:.2f}%".format(loan_pct))

if filtered_df.empty:
    st.warning("No buyers match the selected filters. Adjust the sidebar selections.")
    st.stop()


def count_chart(title, column, top_n=None):
    st.subheader(title)
    counts = filtered_df[column].value_counts()
    if top_n is not None:
        counts = counts.head(top_n)
    if counts.empty:
        st.info("No data available for this chart.")
        return
    counts.index.name = column.replace("_", " ").title()
    st.bar_chart(counts)
    result = counts.rename("Buyer Count").reset_index()
    result["Percentage"] = (result["Buyer Count"] / len(filtered_df) * 100).round(2)
    st.dataframe(result, use_container_width=True)


st.header("Buyer Behavior")
left, right = st.columns(2)
with left:
    count_chart("Buyer Segment Distribution", "buyer_segment")
with right:
    count_chart("Acquisition Purpose Distribution", "acquisition_purpose")
count_chart("Referral Channel Analysis", "referral_channel")

st.header("Geographic Analysis")
count_chart("Top 10 Countries", "country", top_n=10)

st.header("Segment Profiling")
segment_satisfaction = (filtered_df.dropna(subset=["buyer_segment"])
                        .groupby("buyer_segment")["satisfaction_score"].mean().round(2))
st.subheader("Average Satisfaction by Buyer Segment")
if not segment_satisfaction.empty:
    st.bar_chart(segment_satisfaction)
    st.dataframe(segment_satisfaction.rename("Average Satisfaction").reset_index(), use_container_width=True)

st.subheader("Loan Application Rate by Buyer Segment")
loan_rate = (filtered_df.dropna(subset=["buyer_segment"])
             .groupby("buyer_segment")["loan_applied"]
             .apply(lambda s: s.astype(str).str.strip().str.lower().eq("yes").mean() * 100)
             .round(2))
if not loan_rate.empty:
    st.bar_chart(loan_rate)
    st.dataframe(loan_rate.rename("Loan Application Rate (%)").reset_index(), use_container_width=True)

st.header("Buyer Records")
search_text = st.text_input("Search by Client ID, First Name or Last Name", key="buyer_search_input").strip()
searchable_df = filtered_df.copy()
if search_text:
    search_columns = searchable_df[["client_id", "first_name", "last_name"]].fillna("").astype(str)
    matches = search_columns.apply(lambda s: s.str.contains(search_text, case=False, regex=False)).any(axis=1)
    searchable_df = searchable_df.loc[matches]
st.write("Showing {:,} matching buyer records.".format(len(searchable_df)))
st.dataframe(searchable_df, use_container_width=True)
st.download_button(
    label="Download Buyer Records (CSV)",
    data=searchable_df.to_csv(index=False).encode("utf-8"),
    file_name="filtered_buyer_records.csv",
    mime="text/csv",
    key="download_buyer_records",
)

# Additional analysis and submission-ready dashboard sections
st.header("Age and Buyer Demographics")
if "age" not in df.columns and "date_of_birth" in df.columns:
    dob = pd.to_datetime(df["date_of_birth"], errors="coerce")
    today = pd.Timestamp.today().normalize()
    df["age"] = today.year - dob.dt.year - ((dob.dt.month > today.month) | ((dob.dt.month == today.month) & (dob.dt.day > today.day)).astype(int))
    filtered_df = filtered_df.copy()
    filtered_dob = pd.to_datetime(filtered_df["date_of_birth"], errors="coerce")
    filtered_df["age"] = today.year - filtered_dob.dt.year - ((filtered_dob.dt.month > today.month) | ((filtered_dob.dt.month == today.month) & (filtered_dob.dt.day > today.day)).astype(int))
if "age" in filtered_df.columns:
    ages = pd.to_numeric(filtered_df["age"], errors="coerce").dropna()
    if not ages.empty:
        st.subheader("Age Distribution")
        age_counts = ages.round().astype(int).value_counts().sort_index()
        st.bar_chart(age_counts)
    age_by_segment = (filtered_df.assign(age=pd.to_numeric(filtered_df["age"], errors="coerce"))
                      .dropna(subset=["buyer_segment", "age"])
                      .groupby("buyer_segment")["age"].mean().round(2))
    if not age_by_segment.empty:
        st.subheader("Average Age by Buyer Segment")
        st.bar_chart(age_by_segment)
        st.dataframe(age_by_segment.rename("Average Age").reset_index(), use_container_width=True)
    st.subheader("Age by Acquisition Purpose")
    age_by_purpose = (filtered_df.assign(age=pd.to_numeric(filtered_df["age"], errors="coerce"))
                      .groupby("acquisition_purpose")["age"].mean().round(2))
    st.bar_chart(age_by_purpose)
else:
    st.info("Age analysis requires an age or date_of_birth column.")

st.header("Acquisition Purpose Insights")
purpose_loan = (filtered_df.groupby("acquisition_purpose")["loan_applied"]
                .apply(lambda s: s.astype(str).str.strip().str.lower().eq("yes").mean() * 100).round(2))
st.subheader("Loan Application Rate by Acquisition Purpose")
st.bar_chart(purpose_loan)
st.dataframe(purpose_loan.rename("Loan Application Rate (%)").reset_index(), use_container_width=True)
purpose_satisfaction = filtered_df.groupby("acquisition_purpose")["satisfaction_score"].mean().round(2)
st.subheader("Average Satisfaction by Acquisition Purpose")
st.bar_chart(purpose_satisfaction)
st.dataframe(purpose_satisfaction.rename("Average Satisfaction").reset_index(), use_container_width=True)

st.header("Download Reports")
kpi_report = pd.DataFrame([{
    "Total Buyers": count,
    "Average Satisfaction": round(float(mean_satisfaction), 2),
    "Investment Buyers (%)": round(float(investment_pct), 2),
    "Loan Application Rate (%)": round(float(loan_pct), 2),
}])
st.download_button("Download Filtered KPI Summary", kpi_report.to_csv(index=False).encode("utf-8"),
                   file_name="filtered_dashboard_kpis.csv", mime="text/csv", key="download_filtered_kpis")
segment_report = (filtered_df.groupby("buyer_segment", dropna=True)
                  .agg(buyers=("client_id", "count"), average_satisfaction=("satisfaction_score", "mean"))
                  .reset_index())
segment_report["average_satisfaction"] = segment_report["average_satisfaction"].round(2)
st.download_button("Download Filtered Segment Summary", segment_report.to_csv(index=False).encode("utf-8"),
                   file_name="filtered_segment_summary.csv", mime="text/csv", key="download_filtered_segments")

# Saved business insights are project-level results; they do not change with filters.
insights_path = BASE_DIR / "dashboard_data" / "business_insights.csv"
if insights_path.is_file():
    st.subheader("Project Business Insights")
    try:
        insights = pd.read_csv(insights_path)
        st.dataframe(insights, use_container_width=True)
        st.download_button("Download Project Business Insights", insights.to_csv(index=False).encode("utf-8"),
                           file_name="business_insights.csv", mime="text/csv", key="download_business_insights")
    except Exception as exc:
        st.warning("Business insights file could not be read: {}".format(exc))

st.header("Project Documentation")
with st.expander("Methodology"):
    st.markdown("""**Workflow:** Load and clean buyer data; prepare categorical and numerical features; encode and scale the clustering features; perform K-Means clustering; profile the resulting clusters; assign descriptive buyer-segment names; analyze investment behavior, financing and satisfaction; and display results in Streamlit.

**Interpretation:** Segment names are descriptive labels assigned after clustering, not predictions of a buyer's future investment behavior. Dashboard filters summarize existing buyer records.""")
with st.expander("Dataset Overview"):
    st.write("Source file: {}".format(source.name))
    st.write("Rows: {:,}; Columns: {:,}".format(df.shape[0], df.shape[1]))
    st.dataframe(pd.DataFrame({"Column": df.columns, "Data Type": df.dtypes.astype(str).values,
                               "Missing Values": df.isna().sum().values}), use_container_width=True)
with st.expander("System Architecture"):
    st.code("Buyer CSV data\n      |\nData cleaning and feature preparation (Jupyter)\n      |\nEncoding + scaling -> K-Means clustering\n      |\nBuyer segment labels + exported CSV\n      |\nStreamlit dashboard -> filters, analysis, downloads", language="text")
with st.expander("Project Completion Checklist"):
    st.markdown("""- [x] Load and explore buyer dataset
- [x] Clean data and prepare features
- [x] Run K-Means and analyze buyer segments
- [x] Profile segments and derive business insights
- [x] Build Streamlit dashboard and interactive filters
- [x] Add buyer search and CSV exports
- [x] Add demographic and acquisition-purpose analyses
- [ ] Verify dashboard against the final exported CSV and capture screenshots for submission
- [ ] Confirm any additional project requirements not covered by the completed notebook""")

