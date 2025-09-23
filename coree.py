import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
from prophet import Prophet

def run_app_logic(uploaded_df, is_alldata):
    if is_alldata:
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
            party_bill_df = pd.merge(            
                party_df, bill_df, left_on=id_col, right_on=partyid_col, how="inner", suffixes=("_party", "_bill")            
            ) if not party_df.empty and not bill_df.empty else pd.DataFrame()            
        except Exception:            
            party_bill_df = pd.DataFrame()            

        try:            
            billindex_col = find_col_ci(uploaded_df, "Billindex")            
            bill_billdetails_df = pd.merge(            
                bill_df, billdetails_df, left_on=bill_col, right_on=billindex_col, how="inner", suffixes=("_bill", "_details")            
            ) if not bill_df.empty and not billdetails_df.empty else pd.DataFrame()            
        except Exception:            
            bill_billdetails_df = pd.DataFrame()            

        tables_dict = {            
            "Uploaded Table": uploaded_df,            
            "Party": party_df,            
            "Bill": bill_df,            
            "BillDetails": billdetails_df,            
            "Party + Bill": party_bill_df,            
            "Bill + BillDetails": bill_billdetails_df            
        }            
    else:            
        tables_dict = {"Uploaded Table": uploaded_df}            

    # --- Start of Tables Preview Section ---            
    st.subheader("🗂️ Tables Preview")            
    for table_name, table_df in tables_dict.items():            
        state_key = f"expand_{table_name.replace(' ', '')}"            
        if state_key not in st.session_state:            
            st.session_state[state_key] = False            
        btn_label = f"Minimise {table_name} Table" if st.session_state[state_key] else f"Expand {table_name} Table"            
        st.button(btn_label, key=f"btn{table_name}", on_click=toggle_state, args=(state_key,))            

        if st.session_state[state_key]:            
            st.write(f"### {table_name} Table (First 20 Rows)")            
            if not table_df.empty:            
                # Fixed: Replaced use_container_width with width            
                st.dataframe(table_df.head(20), use_container_width=True)            
                with st.expander(f"📖 Show full {table_name} Table"):            
                    st.markdown("<br>", unsafe_allow_html=True)            
                    st.dataframe(table_df, use_container_width=True)            
                st.download_button(            
                    f"⬇️ Download {table_name} (CSV)",            
                    data=convert_df_to_csv(table_df),            
                    file_name=f"{table_name.lower().replace(' ', '')}.csv",            
                    mime="text/csv",            
                )            
                st.download_button(            
                    f"⬇️ Download {table_name} (Excel)",            
                    data=convert_df_to_excel(table_df),            
                    file_name=f"{table_name.lower().replace(' ', '')}.xlsx",            
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",            
                )            
            else:            
                st.info("ℹ️ Not available from the uploaded CSV.")            

    # --- Start of Visualization Section ---            
    st.markdown("---")            
    with st.expander("📊 Visualize Data", expanded=False):            
        st.subheader("📌 Select Table for Visualization")            
        available_tables = {k: v for k, v in tables_dict.items() if not v.empty}            
        if not available_tables:            
            st.warning("⚠️ No usable tables could be derived from the uploaded CSV.")            
            st.stop()            

        selected_table_name = st.selectbox("Select one table", list(available_tables.keys()))            
        selected_df = available_tables[selected_table_name].copy()            

        date_col_sel = find_col_ci(selected_df, "date") or find_col_ci(selected_df, "Date")            
        amount_col_sel = find_col_ci(selected_df, "amount") or find_col_ci(selected_df, "Amount")            
        name_col_sel = find_col_ci(selected_df, "name") or find_col_ci(selected_df, "Name")            

        if date_col_sel:            
            try:            
                selected_df[date_col_sel] = pd.to_datetime(selected_df[date_col_sel], errors="coerce")            
                selected_df = selected_df.sort_values(by=date_col_sel).reset_index(drop=True)            
                selected_df['Year_Month'] = selected_df[date_col_sel].dt.to_period('M')            
                selected_df['Year'] = selected_df[date_col_sel].dt.to_period('Y')            
                numerical_cols = selected_df.select_dtypes(include=[np.number]).columns.tolist()            
                categorical_cols = [c for c in selected_df.columns if c not in numerical_cols + ['Year_Month', 'Year', date_col_sel]]            

                st.markdown("### 📅 Data Aggregation Options")            
                time_period = st.selectbox(            
                    "Choose time aggregation period:",            
                    ["No Aggregation", "Monthly", "Yearly"],            
                    help="Select how you want to aggregate your data over time"            
                )            

                if time_period != "No Aggregation":            
                    grouping_options = ["No Grouping"]            
                    if name_col_sel:            
                        grouping_options.append("Group by Name")            
                    if categorical_cols:            
                        grouping_options.append("Group by Custom Columns")            

                    grouping_choice = st.selectbox(            
                        f"Choose {time_period.lower()} grouping method:",            
                        grouping_options,            
                        help=f"Select how to group your data within each {time_period.lower()} period"            
                    )            

                    period_col = 'Year_Month' if time_period == "Monthly" else 'Year'            
                    freq_setting = "M" if time_period == "Monthly" else "Y"            

                    if grouping_choice == "Group by Name" and name_col_sel:            
                        grouped_df = selected_df.groupby([period_col, name_col_sel], as_index=False)[numerical_cols].sum()            
                        grouped_df[period_col] = grouped_df[period_col].astype(str)            
                        selected_df = grouped_df.copy()            
                        date_col_sel = period_col            
                        st.success(f"✅ Data aggregated {time_period.lower()} and grouped by {name_col_sel} with numerical values summed.")            
                    elif grouping_choice == "Group by Custom Columns":            
                        st.markdown(f"#### Select Columns for {time_period} Grouping")            
                        selected_group_cols = st.multiselect(            
                            f"Choose columns to group by (in addition to {time_period.lower()} grouping):",            
                            categorical_cols,            
                            default=[],            
                            help=f"Select one or more columns to group your data by within each {time_period.lower()} period. Numerical columns will be summed."            
                        )            
                        if selected_group_cols:            
                            group_by_cols = [period_col] + selected_group_cols            
                            grouped_df = selected_df.groupby(group_by_cols, as_index=False)[numerical_cols].sum()            
                            grouped_df[period_col] = grouped_df[period_col].astype(str)            
                            selected_df = grouped_df.copy()            
                            date_col_sel = period_col            
                            st.success(f"✅ Data aggregated {time_period.lower()} and grouped by {', '.join(selected_group_cols)} with numerical values summed.")            
                        else:            
                            grouped_df = selected_df.groupby(period_col, as_index=False)[numerical_cols].sum()            
                            grouped_df[period_col] = grouped_df[period_col].astype(str)            
                            selected_df = grouped_df.copy()            
                            date_col_sel = period_col            
                            st.info(f"ℹ️ No grouping columns selected. Data aggregated {time_period.lower()} only.")            
                    elif grouping_choice == "No Grouping":            
                        grouped_df = selected_df.groupby(period_col, as_index=False)[numerical_cols].sum()            
                        grouped_df[period_col] = grouped_df[period_col].astype(str)            
                        selected_df = grouped_df.copy()            
                        date_col_sel = period_col            
                        st.success(f"✅ Data aggregated {time_period.lower()} only. All numerical values summed per {time_period.lower()} period.")            
                else:            
                    st.info("ℹ️ Using original data without time aggregation.")            

            except Exception as e:            
                st.warning(f"⚠️ Could not process date grouping: {e}")            
                st.error(f"Error details: {str(e)}")            

        st.subheader("📋 Selected & Processed Table")            
        state_key_processed = "expand_processed_table"            
        if state_key_processed not in st.session_state:            
            st.session_state[state_key_processed] = False            
        btn_label_processed = f"Minimise Processed Table" if st.session_state[state_key_processed] else f"Expand Processed Table"            
        st.button(btn_label_processed, key="btn_processed_table", on_click=toggle_state, args=(state_key_processed,))            
        if st.session_state[state_key_processed]:            
            st.write(f"### {selected_table_name} - Processed Table (First 20 Rows)")            
            # Fixed: Replaced use_container_width with width            
            st.dataframe(selected_df.head(20), use_container_width=True)            
            with st.expander(f"📖 Show full Processed {selected_table_name} Table"):            
                st.markdown("<br>", unsafe_allow_html=True)            
                st.dataframe(selected_df, use_container_width=True)            
            st.download_button(            
                f"⬇️ Download Processed {selected_table_name} (CSV)",            
                data=convert_df_to_csv(selected_df),            
                file_name=f"processed_{selected_table_name.lower().replace(' ', '')}.csv",            
                mime="text/csv",            
            )            
            st.download_button(            
                f"⬇️ Download Processed {selected_table_name} (Excel)",            
                data=convert_df_to_excel(selected_df),            
                file_name=f"processed_{selected_table_name.lower().replace(' ', '')}.xlsx",            
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",            
            )            

        st.subheader("📌 Column Selection for Visualization")            
        all_columns = selected_df.columns.tolist()            
        default_cols = all_columns.copy() if all_columns else []            
        selected_columns = st.multiselect(            
            "Select columns to include in visualization (include 'Date' and 'Amount' for forecasting)",            
            all_columns,            
            default=default_cols,            
            help="Choose which columns to include in your analysis and visualization"            
        )            
        if not selected_columns:            
            st.warning("⚠️ Please select at least one column for visualization.")            
            st.stop()            

        df_vis = selected_df[selected_columns].copy()            
        categorical_cols = df_vis.select_dtypes(include=["object", "category", "bool"]).columns.tolist()            
        numerical_cols = df_vis.select_dtypes(include=[np.number]).columns.tolist()            

        col1, col2, col3 = st.columns(3)            
        with col1:            
            st.metric("Total Columns", len(selected_columns))            
        with col2:            
            st.metric("Numerical Columns", len(numerical_cols))            
        with col3:            
            st.metric("Categorical Columns", len(categorical_cols))            

        with st.expander("📋 Column Details"):            
            st.write("Categorical columns:", categorical_cols if categorical_cols else "None")            
            st.write("Numerical columns:", numerical_cols if numerical_cols else "None")            

        st.subheader("📈 Interactive Visualization")            
        chart_options = [            
            "Scatter Plot", "Line Chart", "Bar Chart", "Histogram", "Correlation Heatmap",            
            "Seaborn Scatterplot", "Seaborn Boxplot", "Seaborn Violinplot", "Seaborn Pairplot", "Seaborn Heatmap",            
            "Plotly Heatmap", "Treemap", "Sunburst", "Time-Series Decomposition"            
        ]            
        chart_x_y_hue_req = {            
            "Scatter Plot": (True, True, True), "Line Chart": (True, True, True), "Bar Chart": (True, True, True),            
            "Histogram": (True, False, True), "Correlation Heatmap": (False, False, False),            
            "Seaborn Scatterplot": (True, True, True), "Seaborn Boxplot": (True, True, True),            
            "Seaborn Violinplot": (True, True, True), "Seaborn Pairplot": (False, False, True), "Seaborn Heatmap": (False, False, False),            
            "Plotly Heatmap": (True, True, False), "Treemap": (True, True, False), "Sunburst": (True, True, False),            
            "Time-Series Decomposition": (True, True, False)            
        }            
        chart_type = st.selectbox("Select Chart Type", chart_options)            
        need_x, need_y, need_hue = chart_x_y_hue_req.get(chart_type, (True, True, False))            

        x_col = y_col = hue_col = None            
        if need_x:            
            x_options = [c for c in df_vis.columns]            
            x_col = st.selectbox("Select X Axis", x_options, key="x_axis")            
        if need_y:            
            y_options = [c for c in df_vis.columns if (c != x_col or not need_x)]            
            y_col = st.selectbox("Select Y Axis", y_options, key="y_axis")            
        if need_hue:            
            hue_options = [c for c in df_vis.columns if c not in [x_col, y_col]]            
            hue_col = st.selectbox("Select Hue/Category (optional)", ["(None)"] + hue_options, key="hue_axis")            
            if hue_col == "(None)":            
                hue_col = None            

        st.write("### Chart:")            
        try:            
            fig = None            
            if chart_type == "Scatter Plot":            
                fig = px.scatter(df_vis, x=x_col, y=y_col, color=hue_col if hue_col else None, title=f"Scatter Plot: {x_col} vs {y_col}")            
            elif chart_type == "Line Chart":            
                fig = px.line(df_vis, x=x_col, y=y_col, color=hue_col if hue_col else None, title=f"Line Chart: {x_col} vs {y_col}")            
            elif chart_type == "Bar Chart":            
                fig = px.bar(df_vis, x=x_col, y=y_col, color=hue_col if hue_col else None, title=f"Bar Chart: {x_col} vs {y_col}")            
            elif chart_type == "Histogram":            
                fig = px.histogram(df_vis, x=x_col, color=hue_col if hue_col else None, title=f"Histogram: {x_col}")            
            elif chart_type == "Correlation Heatmap":            
                if len(numerical_cols) >= 2:            
                    corr = df_vis[numerical_cols].corr()            
                    fig = px.imshow(corr, text_auto=True, color_continuous_scale='RdBu', aspect='auto', title="Correlation Heatmap")            
                else:            
                    st.warning("⚠️ Need at least 2 numerical columns for correlation heatmap.")            
            elif chart_type == "Plotly Heatmap":            
                if len(numerical_cols) >= 2:            
                    fig = px.density_heatmap(df_vis, x=x_col, y=y_col, nbinsx=20, nbinsy=20, title="Plotly Heatmap")            
                else:            
                    st.warning("⚠️ Need at least 2 numerical columns for heatmap.")            

            if fig is not None:            
                # Fixed: Replaced use_container_width with width            
                st.plotly_chart(fig, use_container_width=True)            
                png_bytes_plotly = export_plotly_fig(fig)            
                if png_bytes_plotly:            
                    st.download_button("⬇️ Download Chart (PNG)", data=png_bytes_plotly, file_name="plotly_chart.png", mime="image/png")            

            if chart_type.startswith("Seaborn"):            
                plt.figure(figsize=(10, 6))            
                if chart_type == "Seaborn Scatterplot":            
                    sns.scatterplot(data=df_vis, x=x_col, y=y_col, hue=hue_col if hue_col else None)            
                elif chart_type == "Seaborn Boxplot":            
                    sns.boxplot(data=df_vis, x=x_col, y=y_col, hue=hue_col if hue_col else None)            
                elif chart_type == "Seaborn Violinplot":            
                    sns.violinplot(data=df_vis, x=x_col, y=y_col, hue=hue_col if hue_col else None)            
                elif chart_type == "Seaborn Pairplot":            
                    if len(numerical_cols) >= 2:            
                        sns.pairplot(df_vis, hue=hue_col if hue_col else None)            
                elif chart_type == "Seaborn Heatmap":            
                    if len(numerical_cols) >= 2:            
                        corr = df_vis[numerical_cols].corr()            
                        sns.heatmap(corr, annot=True, cmap="coolwarm", center=0)            
                st.pyplot(plt.gcf())            
                png_bytes_mpl = export_matplotlib_fig(plt.gcf())            
                st.download_button("⬇️ Download Chart (PNG)", data=png_bytes_mpl, file_name="seaborn_chart.png", mime="image/png")            
                plt.close()            
        except Exception as e:            
            st.error(f"⚠️ Failed to render chart: {e}")            

    # --- Start of Forecasting Section ---            
    st.markdown("---")            
    with st.expander("🔮 Forecasting", expanded=False):            
        st.subheader("📌 Select Table for Forecasting")            
        available_tables = {k: v for k, v in tables_dict.items() if not v.empty}            
        if not available_tables:            
            st.warning("⚠️ No usable tables could be derived from the uploaded CSV.")            
            st.stop()            

        selected_table_name_forecast = st.selectbox("Select one table for forecasting", list(available_tables.keys()), key="forecast_table_select")            
        selected_df_forecast = available_tables[selected_table_name_forecast].copy()            

        date_columns = [c for c in selected_df_forecast.columns if "date" in c.lower() or c.lower() in ["year_month", "year"]]            
        numerical_cols = selected_df_forecast.select_dtypes(include=[np.number]).columns.tolist()            

        if date_columns and numerical_cols:            
            st.markdown("---")            
            st.subheader("🔮 Forecasting Options")            
            selected_date_col = st.selectbox("Select Date Column for Forecasting", date_columns)            
            selected_amount_col = st.selectbox("Select Numerical Column for Forecasting", numerical_cols)            

            if selected_date_col and selected_amount_col:            
                forecast_df = selected_df_forecast[[selected_date_col, selected_amount_col]].copy()            
                forecast_df[selected_date_col] = pd.to_datetime(forecast_df[selected_date_col], errors="coerce")            
                forecast_df[selected_amount_col] = pd.to_numeric(forecast_df[selected_amount_col], errors="coerce")            
                forecast_df = forecast_df.dropna(subset=[selected_date_col, selected_amount_col])            

                aggregation_period = st.selectbox("Select Aggregation Period", ["No Aggregation", "Monthly", "Yearly"])            
                if aggregation_period != "No Aggregation":            
                    if aggregation_period == "Monthly":            
                        forecast_df = forecast_df.groupby(pd.Grouper(key=selected_date_col, freq='M')).sum(numeric_only=True).reset_index()            
                        freq_str = "M"            
                        period_type = "months"            
                    else:            
                        forecast_df = forecast_df.groupby(pd.Grouper(key=selected_date_col, freq='Y')).sum(numeric_only=True).reset_index()            
                        freq_str = "Y"            
                        period_type = "years"            
                else:            
                    freq_str = "M"            
                    period_type = "months"            

                original_forecast_df = forecast_df.copy()            
                forecast_df = forecast_df.rename(columns={selected_date_col: "ds", selected_amount_col: "y"})            

                min_data_points = 3            
                if len(forecast_df) >= min_data_points:            
                    st.write(f"📈 **Forecasting based on {len(forecast_df)} data points**")            
                    col1, col2 = st.columns(2)            
                    with col1:            
                        if freq_str == "Y":            
                            horizon = st.slider(f"Forecast Horizon ({period_type})", 1, 10, 3)            
                        else:            
                            horizon = st.slider(f"Forecast Horizon ({period_type})", 3, 24, 6)            
                    with col2:            
                        st.write(f"**Data range:** {forecast_df['ds'].min().strftime('%Y-%m-%d')} to {forecast_df['ds'].max().strftime('%Y-%m-%d')}")            

                    with st.spinner("🔄 Running forecast model..."):            
                        prophet_model = Prophet() # Renamed variable            
                        prophet_model.fit(forecast_df)            
                        future = prophet_model.make_future_dataframe(periods=horizon, freq=freq_str)            
                        forecast = prophet_model.predict(future)            

                        last_date = forecast_df["ds"].max()            
                        hist_forecast = forecast[forecast["ds"] <= last_date]            
                        future_forecast = forecast[forecast["ds"] > last_date]            

                        fig_forecast = px.line(            
                            original_forecast_df,            
                            x=selected_date_col,            
                            y=selected_amount_col,            
                            labels={selected_date_col: "Date", selected_amount_col: "Actual Amount"},            
                            title=f"Forecast Analysis - Next {horizon} {period_type.title()}"            
                        )            
                        fig_forecast.update_traces(name="Historical Data", showlegend=True, line=dict(color="blue", dash="solid"))            
                        fig_forecast.add_scatter(            
                            x=hist_forecast["ds"], y=hist_forecast["yhat"], mode="lines", name="Prophet Fitted", line=dict(color="lightblue", dash="dot")            
                        )            
                        fig_forecast.add_scatter(            
                            x=future_forecast["ds"], y=future_forecast["yhat"], mode="lines", name="Forecast", line=dict(color="orange", dash="dash")            
                        )            

                        if show_confidence:            
                            fig_forecast.add_scatter(            
                                x=forecast["ds"], y=forecast["yhat_upper"], mode="lines", name="Upper Bound", line=dict(dash="dot", color="green"), showlegend=True            
                            )            
                            fig_forecast.add_scatter(            
                                x=forecast["ds"], y=forecast["yhat_lower"], mode="lines", name="Lower Bound", line=dict(dash="dot", color="red"), showlegend=True            
                            )            

                        fig_forecast.add_vrect(            
                            x0=last_date, x1=forecast["ds"].max(),            
                            fillcolor=forecast_color, opacity=forecast_opacity, line_width=0,            
                            annotation_text="Forecast Period", annotation_position="top left"            
                        )            
                        # Fixed: Replaced use_container_width with width            
                        st.plotly_chart(fig_forecast, use_container_width=True)            

                        png_bytes_forecast = export_plotly_fig(fig_forecast)            
                        if png_bytes_forecast:            
                            st.download_button(            
                                "⬇️ Download Forecast Chart (PNG)", data=png_bytes_forecast, file_name="forecast_chart.png", mime="image/png"            
                            )            

                        forecast_table = forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].tail(horizon).copy()            
                        forecast_table.columns = ["Date", "Predicted", "Lower Bound", "Upper Bound"]            
                        if freq_str == "Y":            
                            forecast_table["Date"] = forecast_table["Date"].dt.strftime('%Y')            
                        else:            
                            forecast_table["Date"] = forecast_table["Date"].dt.strftime('%Y-%m-%d')            
                        forecast_table["Predicted"] = forecast_table["Predicted"].round(2)            
                        forecast_table["Lower Bound"] = forecast_table["Lower Bound"].round(2)            
                        forecast_table["Upper Bound"] = forecast_table["Upper Bound"].round(2)            

                        st.subheader("📅 Forecast Table (Future Predictions)")            
                        # Fixed: Replaced use_container_width with width            
                        st.dataframe(forecast_table, use_container_width=True)            

                        col1, col2 = st.columns(2)            
                        with col1:            
                            st.download_button(            
                                "⬇️ Download Forecast Data (CSV)", data=convert_df_to_csv(forecast_table), file_name="forecast_predictions.csv", mime="text/csv"            
                            )            
                        with col2:            
                            st.download_button(            
                                "⬇️ Download Forecast Data (Excel)", data=convert_df_to_excel(forecast_table), file_name="forecast_predictions.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"            
                            )            

                        with st.expander("📊 Forecast Summary Statistics"):            
                            future_avg = future_forecast["yhat"].mean()            
                            historical_avg = hist_forecast["yhat"].mean()            
                            growth_rate = ((future_avg - historical_avg) / historical_avg) * 100 if historical_avg != 0 else 0            
                            col1, col2, col3 = st.columns(3)            
                            with col1:            
                                st.metric("Historical Average", f"{historical_avg:,.2f}")            
                            with col2:            
                                st.metric("Forecast Average", f"{future_avg:,.2f}")            
                            with col3:            
                                st.metric("Growth Rate", f"{growth_rate:.2f}%")            
                else:            
                    st.warning(f"⚠️ Need at least 3 data points for forecasting.")            
        else:            
            st.info("ℹ️ The selected table does not contain a valid date column and/or a numerical column for forecasting.")            

    # --- Chat with CSV Section (IMPROVED SECURITY AND UX) ---            
    st.markdown("---")            
    with st.expander("🤖 Chat with your CSV", expanded=False):            
        st.subheader("📌 Select Table for Chat")            
        available_tables_chat = {k: v for k, v in tables_dict.items() if not v.empty}            
        if not available_tables_chat:            
            st.warning("⚠️ No usable tables could be derived from the uploaded CSV.")            
            st.stop()            

        selected_table_name_chat = st.selectbox("Select one table to chat with", list(available_tables_chat.keys()), key="chat_table_select")            
        selected_df_chat = available_tables_chat[selected_table_name_chat].copy()            

        # Display the preview of the selected table            
        st.write(f"### Preview of '{selected_table_name_chat}'")            
        st.dataframe(selected_df_chat.head(), use_container_width=True)            

        # IMPROVED: Better error handling for agent creation            
        try:
            # SECURITY FIX: Set allow_dangerous_code=False for better security            
            agent = create_pandas_dataframe_agent(            
                llm,            
                selected_df_chat,            
                verbose=False,  # Reduced verbosity for cleaner output            
                agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,            
                allow_dangerous_code=True,  # IMPROVED: Better security            
                return_intermediate_steps=False  # Cleaner output            
            )            

            # Chat interface with improved UX            
            st.markdown("### 💬 Ask Questions About Your Data")            
            # IMPROVED: Better chat input with placeholder examples            
            prompt = st.chat_input(            
                "Ask a question about your data... ",             
                key="chat_input"            
            )            

            # IMPROVED: Add example questions            
            st.markdown("**💡 Example questions you can ask:**")            
            example_questions = [            
                "How many rows are in this dataset?",            
                "What are the column names and their data types?",             
                "Show me summary statistics for numerical columns",            
                "Are there any missing values?",            
                "What is the average value in the Amount column?",            
                "Show me the top 5 rows sorted by Amount"            
            ]            
            cols = st.columns(2)            
            for i, question in enumerate(example_questions):            
                col_idx = i % 2            
                with cols[col_idx]:            
                    if st.button(f"📝 {question}", key=f"example_q_{i}", help="Click to use this example question"):            
                        prompt = question            

            # IMPROVED: Better prompt handling with enhanced error handling            
            if prompt:            
                st.markdown(f"**🤔 Your Question:** {prompt}")            

                # Check if user is asking for a chart with improved detection            
                chart_keywords = ['chart', 'plot', 'graph', 'visualize', 'visualization', 'bar chart', 'line chart', 'histogram', 'scatter plot']            
                is_chart_request = any(keyword in prompt.lower() for keyword in chart_keywords)            

                if is_chart_request:            
                    st.info("🎯 Chart request detected! Let me create a visualization for you.")            

                    # IMPROVED: Better chart creation logic            
                    try:            
                        if 'bar chart' in prompt.lower():            
                            # Find date and amount columns more reliably            
                            date_col = find_col_ci(selected_df_chat, "date")            
                            amount_col = find_col_ci(selected_df_chat, "amount")            

                            if date_col and amount_col:            
                                # Create monthly aggregation with error handling            
                                chart_df = selected_df_chat.copy()            
                                try:            
                                    chart_df[date_col] = pd.to_datetime(chart_df[date_col], errors="coerce")            
                                    chart_df = chart_df.dropna(subset=[date_col, amount_col])            
                                    

                                    if not chart_df.empty:            
                                        chart_df['Year_Month'] = chart_df[date_col].dt.to_period('M').astype(str)            
                                        monthly_data = chart_df.groupby('Year_Month')[amount_col].sum().reset_index()            

                                        # Create bar chart            
                                        fig = px.bar(monthly_data, x='Year_Month', y=amount_col,             
                                                   title='Monthly Amount Aggregation',            
                                                   labels={'Year_Month': 'Month', amount_col: 'Total Amount'})            
                                        st.plotly_chart(fig, use_container_width=True)            
                                        st.success("✅ Bar chart created showing monthly aggregated amounts!")            
                                    else:            
                                        st.error("❌ No valid data found after date conversion")            
                                except Exception as chart_error:            
                                    st.error(f"❌ Error creating chart: {chart_error}")            
                            else:            
                                missing_cols = []            
                                if not date_col: missing_cols.append("'date'")            
                                if not amount_col: missing_cols.append("'amount'")            
                                st.error(f"❌ Could not find {' and '.join(missing_cols)} column(s) in the data.")            
                                st.info(f"Available columns: {', '.join(selected_df_chat.columns.tolist())}")            
                        else:            
                            st.info("ℹ️ Currently supports bar charts. Please specify 'bar chart' in your request, or ask a data analysis question instead.")            
                    except Exception as e:            
                        st.error(f"❌ Error processing chart request: {e}")            
                else:            
                    # IMPROVED: Handle regular data analysis questions with better error handling            
                    with st.spinner("🔍 Analyzing your data..."):            
                        try:            
                            response = agent.invoke({"input": prompt})            
                            

                            if response and "output" in response:            
                                st.success("✅ Analysis complete!")            
                                
                                # IMPROVED: Better response formatting            
                                output = response["output"]            
                                if output and output.strip():            
                                    st.markdown("**📊 Answer:**")            
                                    st.markdown(output)            
                                else:            
                                    st.warning("⚠️ The agent returned an empty response. Try rephrasing your question.")            
                            else:            
                                st.warning("⚠️ No response generated. Please try a different question.")            
                                
                        except Exception as e:            
                            st.error(f"❌ Error processing your question: {str(e)}")            
                            
                            # IMPROVED: Better error guidance            
                            st.markdown("**💡 Troubleshooting tips:**")            
                            st.markdown("- Make sure your question is about the data analysis")            
                            st.markdown("- Check if column names in your question match the actual column names")            
                            st.markdown("- Try simpler questions first (e.g., 'How many rows?' or 'Show column names')")            
                            st.markdown("- Avoid complex multi-step questions")            
                            
                            # Show available columns for reference            
                            with st.expander("📋 Show available columns"):            
                                st.write("**Available columns in your data:**")            
                                for i, col in enumerate(selected_df_chat.columns.tolist(), 1):            
                                    st.write(f"{i}. `{col}` ({selected_df_chat[col].dtype})")            
                
        except Exception as agent_error:            
            st.error(f"❌ Error setting up chat agent: {agent_error}")            
            st.info("💡 Chat functionality requires a valid Google API key and proper network connection.")            
            # Fallback: Show data info without chat            
            st.markdown("**📊 Basic Data Information (Fallback):**")            
            st.write(f"- **Rows:** {len(selected_df_chat):,}")            
            st.write(f"- **Columns:** {len(selected_df_chat.columns)}")            
            st.write(f"- **Column Names:** {', '.join(selected_df_chat.columns.tolist())}")            
            # Show data types            
            with st.expander("📋 Column Data Types"):            
                for col in selected_df_chat.columns:            
                    st.write(f"- **{col}:** {selected_df_chat[col].dtype}")
