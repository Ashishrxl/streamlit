import streamlit as st
import pandas as pd

def handle_renaming_flow(uploaded_file, run_app_logic, llm, forecast_color, forecast_opacity, show_confidence):
    st.warning("⚠️ Please confirm your CSV file structure.")
    st.subheader("📋 Confirm File Structure")
    header_option = st.radio("Does your CSV file have a header row?", ["Yes", "No"])

    if header_option == "Yes":
        try:
            uploaded_df = pd.read_csv(uploaded_file, low_memory=False)
            st.success("✅ File loaded with header successfully!")
            st.info("Now, please confirm the column names for analysis.")
            col_confirm = st.radio("Are the column names correct?", ["Yes", "No, I want to rename them"])
            if col_confirm == "Yes":
                st.success("Column names confirmed!")
                # Save df and llm into session state for multipage access
                st.session_state['uploaded_df'] = uploaded_df
                if llm is not None:
                    st.session_state['llm'] = llm
                if forecast_color is not None:
                    st.session_state['forecast_color'] = forecast_color
                if forecast_opacity is not None:
                    st.session_state['forecast_opacity'] = forecast_opacity
                if show_confidence is not None:
                    st.session_state['show_confidence'] = show_confidence
                st.info("✅ You can now navigate to Visualization or Chat pages.")
            else:
                st.info("Please provide the new column names.")
                new_cols_dict = {}
                original_cols = uploaded_df.columns.tolist()
                with st.form("column_rename_form"):
                    st.write("### Rename Columns")
                    for i, col in enumerate(original_cols):
                        new_name = st.text_input(f"Rename '{col}' to:", value=col, key=f"rename_{i}")
                        new_cols_dict[col] = new_name if new_name.strip() else col
                    submitted = st.form_submit_button("✅ Apply Column Renames", type="primary")
                    if submitted:
                        try:
                            uploaded_df = uploaded_df.rename(columns=new_cols_dict)
                            st.success("✅ Columns renamed successfully!")
                            st.info("**Updated column names:**")
                            st.write(list(uploaded_df.columns))
                            # Save renamed df and context to session state
                            st.session_state['uploaded_df'] = uploaded_df
                            if llm is not None:
                                st.session_state['llm'] = llm
                            if forecast_color is not None:
                                st.session_state['forecast_color'] = forecast_color
                            if forecast_opacity is not None:
                                st.session_state['forecast_opacity'] = forecast_opacity
                            if show_confidence is not None:
                                st.session_state['show_confidence'] = show_confidence
                            st.info("✅ You can now navigate to Visualization or Chat pages.")
                        except Exception as rename_error:
                            st.error(f"❌ Error renaming columns: {rename_error}")
        except Exception as e:
            st.error(f"❌ Error reading CSV: {e}")
            st.stop()
    else:
        try:
            uploaded_df = pd.read_csv(uploaded_file, header=None, low_memory=False)
            st.success("✅ File loaded without header successfully!")
            st.warning("⚠️ Auto-generating column names...")
            num_cols = len(uploaded_df.columns)
            default_cols = [f"Column_{i+1}" for i in range(num_cols)]
            uploaded_df.columns = default_cols
            st.info(f"📊 Generated {num_cols} column names: {', '.join(default_cols[:5])}{'...' if num_cols > 5 else ''}")
            if st.checkbox("🏷️ Customize column names"):
                with st.form("auto_column_rename"):
                    st.write("### Customize Auto-Generated Column Names")
                    custom_cols = {}
                    for i, col in enumerate(default_cols):
                        custom_name = st.text_input(f"Column {i+1} name:", value=col, key=f"custom_{i}")
                        custom_cols[col] = custom_name if custom_name.strip() else col
                    if st.form_submit_button("✅ Apply Custom Names"):
                        uploaded_df = uploaded_df.rename(columns=custom_cols)
                        st.success("✅ Column names updated!")
            # Save df and context to session state
            st.session_state['uploaded_df'] = uploaded_df
            if llm is not None:
                st.session_state['llm'] = llm
            if forecast_color is not None:
                st.session_state['forecast_color'] = forecast_color
            if forecast_opacity is not None:
                st.session_state['forecast_opacity'] = forecast_opacity
            if show_confidence is not None:
                st.session_state['show_confidence'] = show_confidence
            st.info("✅ You can now navigate to Visualization or Chat pages.")
        except Exception as e:
            st.error(f"❌ Error reading CSV without header: {e}")
            st.info("💡 Please check if your file is a valid CSV format.")