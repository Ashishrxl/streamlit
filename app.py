# app.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.io as pio
from prophet import Prophet
import seaborn as sns
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
import io

# Page config
st.set_page_config(page_title="CSV Visualizer & Forecaster", layout="wide")
st.title("📊 CSV Data Visualizer with Forecasting (Interactive)")

# --- Helpers ---
def find_col_ci(df: pd.DataFrame, target: str):
    for c in df.columns:
        if c.lower() == target.lower():
            return c
    return None

def convert_df_to_csv(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")

def convert_df_to_excel(df: pd.DataFrame) -> bytes:
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="xlsxwriter") as writer:
        df.to_excel(writer, index=False, sheet_name="Sheet1")
    return buffer.getvalue()

def export_plotly_fig(fig):
    """Return PNG bytes via kaleido if available, else None."""
    try:
        return pio.to_image(fig, format="png", engine="kaleido")
    except Exception:
        return None

def export_plotly_html(fig):
    return fig.to_html(include_plotlyjs="cdn")

def export_matplotlib_fig(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    buf.seek(0)
    return buf.getvalue()

# Sidebar controls
st.sidebar.header("⚙️ Settings")
forecast_color = st.sidebar.color_picker("Forecast highlight color", "#FFA500")
forecast_opacity = st.sidebar.slider("Forecast highlight opacity", 0.05, 1.0, 0.12, step=0.01)
show_confidence = st.sidebar.checkbox("Show confidence interval (upper/lower bounds)", True)

# File upload
uploaded_file = st.file_uploader("Upload your CSV file (joined table)", type=["csv"])
if uploaded_file is None:
    st.info("Upload a CSV to start. The app will derive tables and let you visualize/forecast.")
    st.stop()

# Read CSV
try:
    uploaded_df = pd.read_csv(uploaded_file, low_memory=False)
except Exception as e:
    st.error(f"❌ Error reading CSV: {e}")
    st.stop()

st.success("✅ File uploaded successfully!")

# Preview uploaded
st.subheader("🔍 Uploaded Table Preview (First 20 Rows)")
st.dataframe(uploaded_df.head(20))
with st.expander("📖 Show full uploaded table"):
    st.dataframe(uploaded_df)

# --- Build derived tables (preserve your earlier logic) ---
id_col = find_col_ci(uploaded_df, "ID")
name_col = find_col_ci(uploaded_df, "Name")
party_df = uploaded_df[[id_col, name_col]].drop_duplicates().reset_index(drop=True) if id_col and name_col else pd.DataFrame()

bill_col = find_col_ci(uploaded_df, "Bill")
partyid_col = find_col_ci(uploaded_df, "PartyId")
date_col_master = find_col_ci(uploaded_df, "Date")
amount_col_master = find_col_ci(uploaded_df, "Amount")
bill_df = (
    uploaded_df[[bill_col, partyid_col, date_col_master, amount_col_master]].drop_duplicates().reset_index(drop=True)
    if bill_col and partyid_col and date_col_master and amount_col_master else pd.DataFrame()
)

billdetails_cols = [find_col_ci(uploaded_df, c) for c in ["IndexId", "Billindex", "Item", "Qty", "Rate", "Less"]]
billdetails_cols = [c for c in billdetails_cols if c]
billdetails_df = uploaded_df[billdetails_cols].drop_duplicates().reset_index(drop=True) if billdetails_cols else pd.DataFrame()

try:
    party_bill_df = pd.merge(party_df, bill_df, left_on=id_col, right_on=partyid_col, how="inner", suffixes=("_party", "_bill")) if not party_df.empty and not bill_df.empty else pd.DataFrame()
except Exception:
    party_bill_df = pd.DataFrame()

try:
    billindex_col = find_col_ci(uploaded_df, "Billindex")
    bill_billdetails_df = pd.merge(bill_df, billdetails_df, left_on=bill_col, right_on=billindex_col, how="inner", suffixes=("_bill", "_details")) if not bill_df.empty and not billdetails_df.empty else pd.DataFrame()
except Exception:
    bill_billdetails_df = pd.DataFrame()

# --- Show tables and provide downloads ---
st.subheader("🗂️ Tables Preview")
tables_dict = {
    "Uploaded Table": uploaded_df,
    "Party": party_df,
    "Bill": bill_df,
    "BillDetails": billdetails_df,
    "Party + Bill": party_bill_df,
    "Bill + BillDetails": bill_billdetails_df
}

for table_name, table_df in tables_dict.items():
    st.write(f"### {table_name} Table (First 20 Rows)")
    if not table_df.empty:
        st.dataframe(table_df.head(20))
        with st.expander(f"📖 Show full {table_name} Table"):
            st.dataframe(table_df)

        st.download_button(
            f"⬇️ Download {table_name} (CSV)",
            data=convert_df_to_csv(table_df),
            file_name=f"{table_name.lower().replace(' ', '_')}.csv",
            mime="text/csv",
        )
        st.download_button(
            f"⬇️ Download {table_name} (Excel)",
            data=convert_df_to_excel(table_df),
            file_name=f"{table_name.lower().replace(' ', '_')}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )
    else:
        st.info("ℹ️ Not available from the uploaded CSV.")

# --- Select table for visualization ---
st.subheader("📌 Select Table for Visualization")
available_tables = {k: v for k, v in tables_dict.items() if not v.empty}
if not available_tables:
    st.warning("⚠️ No usable tables could be derived from the uploaded CSV.")
    st.stop()

selected_table_name = st.selectbox("Select one table", list(available_tables.keys()))
selected_df = available_tables[selected_table_name].copy()
st.write(f"Selected Table: **{selected_table_name}** (First 20 Rows)")
st.dataframe(selected_df.head(20))

# --- Column selection ---
st.subheader("📌 Column Selection for Visualization")
all_columns = selected_df.columns.tolist()
# Default to all columns so user immediately sees charts if possible:
default_cols = all_columns.copy() if all_columns else []
selected_columns = st.multiselect(
    "Select columns to include in visualization (include 'Date' and 'Amount' for forecasting)",
    all_columns,
    default=default_cols
)

if not selected_columns:
    st.warning("⚠️ Please select at least one column for visualization.")
    st.stop()

df_vis = selected_df[selected_columns].copy()
categorical_cols = df_vis.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
numerical_cols = df_vis.select_dtypes(include=[np.number]).columns.tolist()

st.write("**Categorical columns:**", categorical_cols if categorical_cols else "None")
st.write("**Numerical columns:**", numerical_cols if numerical_cols else "None")

# --- Visualization choices (robust) ---
st.subheader("📈 Interactive Visualization")
chart_type = st.selectbox(
    "Select Chart Type",
    [
        "Scatter Plot", "Line Chart", "Bar Chart", "Histogram", "Correlation Heatmap",
        "Seaborn Scatterplot", "Seaborn Boxplot", "Seaborn Violinplot", "Seaborn Pairplot",
        "Seaborn Heatmap", "Plotly Heatmap", "Treemap", "Sunburst", "Time-Series Decomposition"
    ]
)

widget_key_base = selected_table_name.replace(" ", "_")
fig = None
fig_matplotlib = None
chart_displayed = False

# --- Plotly chart helpers for common charts ---
def display_and_mark_plotly(f):
    st.plotly_chart(f, use_container_width=True)
    return True

def display_and_mark_matplotlib(fig_obj):
    st.pyplot(fig_obj)
    plt.close(fig_obj)
    return True

# Scatter (Plotly)
if chart_type == "Scatter Plot":
    if len(numerical_cols) >= 2:
        x_axis = st.selectbox("Select X-axis (numerical)", numerical_cols, key=f"px_scatter_x_{widget_key_base}")
        y_axis = st.selectbox("Select Y-axis (numerical)", numerical_cols, key=f"px_scatter_y_{widget_key_base}")
        color_col = st.selectbox("Color grouping (optional)", ["None"] + categorical_cols, key=f"px_scatter_color_{widget_key_base}")
        color_col = None if color_col == "None" else color_col
        fig = px.scatter(df_vis, x=x_axis, y=y_axis, color=color_col, title=f"{y_axis} vs {x_axis}")
        chart_displayed = display_and_mark_plotly(fig)
    else:
        st.warning("⚠️ Need at least two numerical columns for a scatter plot.")

# Line Chart (Plotly)
elif chart_type == "Line Chart":
    if numerical_cols:
        x_axis = st.selectbox("Select X-axis", df_vis.columns.tolist(), key=f"px_line_x_{widget_key_base}")
        y_axis = st.selectbox("Select Y-axis (numerical)", numerical_cols, key=f"px_line_y_{widget_key_base}")
        color_col = st.selectbox("Color grouping (optional)", ["None"] + categorical_cols, key=f"px_line_color_{widget_key_base}")
        color_col = None if color_col == "None" else color_col
        fig = px.line(df_vis, x=x_axis, y=y_axis, color=color_col, title=f"{y_axis} over {x_axis}")
        chart_displayed = display_and_mark_plotly(fig)
    else:
        st.warning("⚠️ Need at least one numerical column for a line chart.")

# Bar Chart (Plotly)
elif chart_type == "Bar Chart":
    if categorical_cols and numerical_cols:
        x_axis = st.selectbox("Select X-axis (categorical)", categorical_cols, key=f"px_bar_x_{widget_key_base}")
        y_axis = st.selectbox("Select Y-axis (numerical)", numerical_cols, key=f"px_bar_y_{widget_key_base}")
        color_col = st.selectbox("Grouping (optional)", ["None"] + categorical_cols, key=f"px_bar_color_{widget_key_base}")
        color_col = None if color_col == "None" else color_col
        bar_mode = st.radio("Bar Mode", ["Stacked", "Grouped"], horizontal=True, key=f"px_bar_mode_{widget_key_base}")
        fig = px.bar(df_vis, x=x_axis, y=y_axis, color=color_col, barmode="stack" if bar_mode == "Stacked" else "group",
                     title=f"{y_axis} by {x_axis}")
        chart_displayed = display_and_mark_plotly(fig)
    else:
        st.warning("⚠️ Need at least one categorical and one numerical column for a bar chart.")

# Histogram (Plotly)
elif chart_type == "Histogram":
    if numerical_cols:
        hist_col = st.selectbox("Select column for histogram", numerical_cols, key=f"px_hist_{widget_key_base}")
        nbins = st.slider("Number of bins", 5, 100, 30, key=f"px_hist_bins_{widget_key_base}")
        fig = px.histogram(df_vis, x=hist_col, nbins=nbins, title=f"Histogram of {hist_col}")
        chart_displayed = display_and_mark_plotly(fig)
    else:
        st.warning("⚠️ Need at least one numerical column for a histogram.")

# Correlation Heatmap (Plotly by default)
elif chart_type == "Correlation Heatmap":
    if len(numerical_cols) > 1:
        corr = df_vis[numerical_cols].corr()
        fig = px.imshow(corr, text_auto=True, aspect="auto", title="Correlation Matrix")
        chart_displayed = display_and_mark_plotly(fig)
    else:
        st.warning("⚠️ Need more than one numerical column for correlation heatmap.")

# Seaborn Scatterplot (with hue)
elif chart_type == "Seaborn Scatterplot":
    if len(numerical_cols) >= 2:
        x_axis = st.selectbox("Select X-axis (numerical)", numerical_cols, key=f"sns_scatter_x_{widget_key_base}")
        y_axis = st.selectbox("Select Y-axis (numerical)", numerical_cols, key=f"sns_scatter_y_{widget_key_base}")
        hue_col = st.selectbox("Select hue (categorical, optional)", ["None"] + categorical_cols, key=f"sns_scatter_hue_{widget_key_base}")
        hue_col = None if hue_col == "None" else hue_col
        fig_matplotlib, ax = plt.subplots(figsize=(8, 6))
        sns.scatterplot(data=df_vis, x=x_axis, y=y_axis, hue=hue_col, ax=ax)
        ax.set_title(f"{y_axis} vs {x_axis}")
        chart_displayed = display_and_mark_matplotlib(fig_matplotlib)

# Seaborn Boxplot
elif chart_type == "Seaborn Boxplot":
    if categorical_cols and numerical_cols:
        x_axis = st.selectbox("Select categorical X-axis", categorical_cols, key=f"sns_box_x_{widget_key_base}")
        y_axis = st.selectbox("Select numerical Y-axis", numerical_cols, key=f"sns_box_y_{widget_key_base}")
        hue_col = st.selectbox("Select hue (categorical, optional)", ["None"] + categorical_cols, key=f"sns_box_hue_{widget_key_base}")
        hue_col = None if hue_col == "None" else hue_col
        fig_matplotlib, ax = plt.subplots(figsize=(8, 6))
        sns.boxplot(data=df_vis, x=x_axis, y=y_axis, hue=hue_col, ax=ax)
        ax.set_title(f"{y_axis} by {x_axis}")
        chart_displayed = display_and_mark_matplotlib(fig_matplotlib)
    else:
        st.warning("⚠️ Need at least one categorical and one numerical column for boxplot.")

# Seaborn Violinplot
elif chart_type == "Seaborn Violinplot":
    if categorical_cols and numerical_cols:
        x_axis = st.selectbox("Select categorical X-axis", categorical_cols, key=f"sns_violin_x_{widget_key_base}")
        y_axis = st.selectbox("Select numerical Y-axis", numerical_cols, key=f"sns_violin_y_{widget_key_base}")
        hue_col = st.selectbox("Select hue (categorical, optional)", ["None"] + categorical_cols, key=f"sns_violin_hue_{widget_key_base}")
        hue_col = None if hue_col == "None" else hue_col
        fig_matplotlib, ax = plt.subplots(figsize=(8, 6))
        # only use split=True when hue has exactly 2 unique values
        if hue_col and df_vis[hue_col].nunique() == 2:
            sns.violinplot(data=df_vis, x=x_axis, y=y_axis, hue=hue_col, split=True, ax=ax)
        else:
            sns.violinplot(data=df_vis, x=x_axis, y=y_axis, hue=hue_col, ax=ax)
        ax.set_title(f"{y_axis} by {x_axis}")
        chart_displayed = display_and_mark_matplotlib(fig_matplotlib)
    else:
        st.warning("⚠️ Need at least one categorical and one numerical column for violinplot.")

# Seaborn Pairplot
elif chart_type == "Seaborn Pairplot":
    if len(numerical_cols) >= 2:
        hue_col = st.selectbox("Select hue (categorical, optional)", ["None"] + categorical_cols, key=f"sns_pair_hue_{widget_key_base}")
        hue_col = None if hue_col == "None" else hue_col
        # pairplot returns a PairGrid; use .fig to render
        pairplot = sns.pairplot(df_vis[numerical_cols] if hue_col is None else df_vis[numerical_cols + [hue_col]], hue=hue_col)
        chart_displayed = display_and_mark_matplotlib(pairplot.fig)
    else:
        st.warning("⚠️ Need at least two numerical columns for pairplot.")

# Seaborn Heatmap (pivot or corr)
elif chart_type == "Seaborn Heatmap":
    if categorical_cols and numerical_cols:
        row_cat = st.selectbox("Select row categorical", categorical_cols, key=f"sns_heat_row_{widget_key_base}")
        col_cat = st.selectbox("Select column categorical", categorical_cols, key=f"sns_heat_col_{widget_key_base}")
        value_num = st.selectbox("Select value (numerical)", numerical_cols, key=f"sns_heat_val_{widget_key_base}")
        pivot_df = df_vis.pivot_table(index=row_cat, columns=col_cat, values=value_num, aggfunc="mean")
        fig_matplotlib, ax = plt.subplots(figsize=(10, 6))
        sns.heatmap(pivot_df, cmap="coolwarm", annot=True, fmt=".2f", ax=ax)
        ax.set_title(f"{value_num} mean: {row_cat} vs {col_cat}")
        chart_displayed = display_and_mark_matplotlib(fig_matplotlib)
    elif len(numerical_cols) > 1:
        corr = df_vis[numerical_cols].corr()
        fig_matplotlib, ax = plt.subplots(figsize=(9, 7))
        sns.heatmap(corr, cmap="coolwarm", annot=True, fmt=".2f", ax=ax)
        ax.set_title("Correlation heatmap")
        chart_displayed = display_and_mark_matplotlib(fig_matplotlib)
    else:
        st.warning("⚠️ Need categorical + numerical or multiple numerical columns for heatmap.")

# Plotly Heatmap (interactive)
elif chart_type == "Plotly Heatmap":
    if categorical_cols and numerical_cols:
        row_cat = st.selectbox("Select row categorical", categorical_cols, key=f"px_heat_row_{widget_key_base}")
        col_cat = st.selectbox("Select column categorical", categorical_cols, key=f"px_heat_col_{widget_key_base}")
        value_num = st.selectbox("Select value (numerical)", numerical_cols, key=f"px_heat_val_{widget_key_base}")
        pivot_df = df_vis.pivot_table(index=row_cat, columns=col_cat, values=value_num, aggfunc="mean")
        # px.imshow accepts DataFrame directly
        fig = px.imshow(pivot_df, text_auto=True, aspect="auto", title=f"{value_num} mean: {row_cat} vs {col_cat}", color_continuous_scale="Viridis")
        chart_displayed = display_and_mark_plotly(fig)
    elif len(numerical_cols) > 1:
        corr = df_vis[numerical_cols].corr()
        fig = px.imshow(corr, text_auto=True, aspect="auto", title="Correlation matrix", color_continuous_scale="Viridis")
        chart_displayed = display_and_mark_plotly(fig)
    else:
        st.warning("⚠️ Need categorical + numerical or multiple numerical columns for heatmap.")

# Treemap
elif chart_type == "Treemap":
    if categorical_cols and numerical_cols:
        path_cols = st.multiselect("Hierarchy (categorical)", categorical_cols, default=categorical_cols[:1], key=f"treemap_path_{widget_key_base}")
        value_col = st.selectbox("Value (numerical)", numerical_cols, key=f"treemap_val_{widget_key_base}")
        if path_cols:
            fig = px.treemap(df_vis, path=path_cols, values=value_col, title=f"Treemap of {value_col}")
            chart_displayed = display_and_mark_plotly(fig)
        else:
            st.warning("⚠️ Select at least one categorical for hierarchy.")
    else:
        st.warning("⚠️ Need categorical + numerical columns for treemap.")

# Sunburst
elif chart_type == "Sunburst":
    if categorical_cols and numerical_cols:
        path_cols = st.multiselect("Hierarchy (categorical)", categorical_cols, default=categorical_cols[:1], key=f"sunburst_path_{widget_key_base}")
        value_col = st.selectbox("Value (numerical)", numerical_cols, key=f"sunburst_val_{widget_key_base}")
        if path_cols:
            fig = px.sunburst(df_vis, path=path_cols, values=value_col, title=f"Sunburst of {value_col}")
            chart_displayed = display_and_mark_plotly(fig)
        else:
            st.warning("⚠️ Select at least one categorical for hierarchy.")
    else:
        st.warning("⚠️ Need categorical + numerical columns for sunburst.")

# Time-Series Decomposition (Matplotlib)
elif chart_type == "Time-Series Decomposition":
    date_col = find_col_ci(df_vis, "date")
    amount_col = find_col_ci(df_vis, "amount")
    if date_col and amount_col:
        ts_df = df_vis[[date_col, amount_col]].copy()
        ts_df[date_col] = pd.to_datetime(ts_df[date_col], errors="coerce")
        ts_df[amount_col] = pd.to_numeric(ts_df[amount_col], errors="coerce")
        ts_df = ts_df.dropna().set_index(date_col).sort_index()
        if len(ts_df) >= 12:
            model_type = st.radio("Decomposition Model", ["additive", "multiplicative"], horizontal=True, key=f"decomp_model_{widget_key_base}")
            decomposition = seasonal_decompose(ts_df[amount_col], model=model_type, period=12)
            fig_matplotlib, axes = plt.subplots(4, 1, figsize=(12, 8), sharex=True)
            decomposition.observed.plot(ax=axes[0], title="Observed")
            decomposition.trend.plot(ax=axes[1], title="Trend")
            decomposition.seasonal.plot(ax=axes[2], title="Seasonality")
            decomposition.resid.plot(ax=axes[3], title="Residuals")
            plt.tight_layout()
            chart_displayed = display_and_mark_matplotlib(fig_matplotlib)
        else:
            st.warning("⚠️ Need at least 12 time points for decomposition.")
    else:
        st.info("ℹ️ Need 'Date' and 'Amount' columns for decomposition.")

# If a chart was shown, provide export options
if chart_displayed:
    # Plotly export (if fig exists)
    if 'fig' in locals() and fig is not None:
        png_bytes = export_plotly_fig(fig)
        if png_bytes:
            st.download_button(f"⬇️ Download {chart_type} (PNG)", data=png_bytes, file_name=f"{chart_type.replace(' ', '_').lower()}.png", mime="image/png")
        # HTML fallback
        st.download_button(f"⬇️ Download {chart_type} (HTML, interactive)", data=export_plotly_html(fig), file_name=f"{chart_type.replace(' ', '_').lower()}.html", mime="text/html")

    # Matplotlib export (if fig_matplotlib exists)
    if 'fig_matpl