import streamlit as st
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
from langchain.agents import AgentType
from data_utils import find_col_ci

def chat_with_csv_section(tables_dict, llm):
    st.markdown("---")
    with st.expander("🤖 Chat with your CSV", expanded=False):
        st.subheader("📌 Select Table for Chat")
        available_tables_chat = {k: v for k, v in tables_dict.items() if not v.empty}
        if not available_tables_chat:
            st.warning("⚠️ No usable tables could be derived from the uploaded CSV.")
            st.stop()

        selected_table_name_chat = st.selectbox(
            "Select one table to chat with",
            list(available_tables_chat.keys()),
            key="chat_table_select"
        )
        selected_df_chat = available_tables_chat[selected_table_name_chat].copy()

        st.write(f"### Preview of '{selected_table_name_chat}'")
        st.dataframe(selected_df_chat.head(), use_container_width=True)

        try:
            agent = create_pandas_dataframe_agent(
                llm,
                selected_df_chat,
                verbose=False,
                agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
                allow_dangerous_code=True,
                return_intermediate_steps=False
            )

            st.markdown("### 💬 Ask Questions About Your Data")
            prompt = st.chat_input(
                "Ask a question about your data... ",
                key="chat_input"
            )

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

            if prompt:
                st.markdown(f"**🤔 Your Question:** {prompt}")
                chart_keywords = ['chart', 'plot', 'graph', 'visualize', 'visualization', 'bar chart', 'line chart', 'histogram', 'scatter plot']
                is_chart_request = any(keyword in prompt.lower() for keyword in chart_keywords)

                if is_chart_request:
                    st.info("🎯 Chart request detected! Let me create a visualization for you.")
                    try:
                        if 'bar chart' in prompt.lower():
                            date_col = find_col_ci(selected_df_chat, "date")
                            amount_col = find_col_ci(selected_df_chat, "amount")
                            if date_col and amount_col:
                                chart_df = selected_df_chat.copy()
                                try:
                                    chart_df[date_col] = pd.to_datetime(chart_df[date_col], errors="coerce")
                                    chart_df = chart_df.dropna(subset=[date_col, amount_col])
                                    if not chart_df.empty:
                                        chart_df['Year_Month'] = chart_df[date_col].dt.to_period('M').astype(str)
                                        monthly_data = chart_df.groupby('Year_Month')[amount_col].sum().reset_index()
                                        import plotly.express as px
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
                    with st.spinner("🔍 Analyzing your data..."):
                        try:
                            response = agent.invoke({"input": prompt})
                            if response and "output" in response:
                                st.success("✅ Analysis complete!")
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
                            st.markdown("**💡 Troubleshooting tips:**")
                            st.markdown("- Make sure your question is about the data analysis")
                            st.markdown("- Check if column names in your question match the actual column names")
                            st.markdown("- Try simpler questions first (e.g., 'How many rows?' or 'Show column names')")
                            st.markdown("- Avoid complex multi-step questions")
                            with st.expander("📋 Show available columns"):
                                st.write("**Available columns in your data:**")
                                for i, col in enumerate(selected_df_chat.columns.tolist(), 1):
                                    st.write(f"{i}. `{col}` ({selected_df_chat[col].dtype})")

        except Exception as agent_error:
            st.error(f"❌ Error setting up chat agent: {agent_error}")
            st.info("💡 Chat functionality requires a valid Google API key and proper network connection.")
            st.markdown("**📊 Basic Data Information (Fallback):**")
            st.write(f"- **Rows:** {len(selected_df_chat):,}")
            st.write(f"- **Columns:** {len(selected_df_chat.columns)}")
            st.write(f"- **Column Names:** {', '.join(selected_df_chat.columns.tolist())}")
            with st.expander("📋 Column Data Types"):
                for col in selected_df_chat.columns:
                    st.write(f"- **{col}:** {selected_df_chat[col].dtype}")