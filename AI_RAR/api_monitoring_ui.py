import streamlit as st
import requests
import time
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import json

# Configure page settings
st.set_page_config(
    page_title="API Endpoint Dashboard",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Add custom CSS
st.markdown("""
    <style>
    .main {
        background-color: #f5f7f9;
    }
    .status-completed {
        color: green;
        font-weight: bold;
    }
    .status-running {
        color: orange;
        font-weight: bold;
    }
    .status-failed {
        color: red;
        font-weight: bold;
    }
    .status-pending {
        color: #1E90FF;
        font-weight: bold;
    }
    .api-key-display {
        font-family: monospace;
        background-color: #f0f0f0;
        padding: 2px 5px;
        border-radius: 3px;
    }
    </style>
    """, unsafe_allow_html=True)

# Initialize session state
if 'jobs' not in st.session_state:
    st.session_state.jobs = {}
if 'api_keys' not in st.session_state:
    st.session_state.api_keys = {}
if 'api_key_usage' not in st.session_state:
    st.session_state.api_key_usage = {}
if 'request_times' not in st.session_state:
    st.session_state.request_times = {}

def get_masked_api_key(api_key):
    """Return masked API key showing only first 5 and last 3 characters"""
    if len(api_key) <= 8:
        return api_key
    return f"{api_key[:5]}...{api_key[-3:]}"

def update_job_status(trace_id, api_key, base_url):
    """Update job status and track metrics"""
    try:
        start_time = time.time()
        response = requests.get(f"{base_url}/ai/rar/v1/status/{trace_id}")
        end_time = time.time()
        
        request_time = end_time - start_time
        
        if response.status_code == 200:
            job_data = response.json()
            
            # Update job data
            st.session_state.jobs[trace_id] = {
                "trace_id": trace_id,
                "status": job_data["status"],
                "progress": job_data["progress"],
                "results": job_data["results"],
                "api_key": api_key,
                "request_time": request_time,
                "last_updated": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "status_code": response.status_code
            }
            
            # Update API key usage
            masked_key = get_masked_api_key(api_key)
            if masked_key not in st.session_state.api_key_usage:
                st.session_state.api_key_usage[masked_key] = 0
            st.session_state.api_key_usage[masked_key] += 1
            
            # Update request times
            if masked_key not in st.session_state.request_times:
                st.session_state.request_times[masked_key] = []
            st.session_state.request_times[masked_key].append(request_time)
            
            return True
        else:
            st.session_state.jobs[trace_id]["status"] = "failed"
            st.session_state.jobs[trace_id]["status_code"] = response.status_code
            return False
    except Exception as e:
        st.error(f"Error updating job status: {str(e)}")
        return False

def submit_job(job_openings, resumes, api_key, base_url):
    """Submit a job to the API"""
    try:
        headers = {"Authorization": f"Bearer {api_key}"}
        data = {
            "job_openings": job_openings,
            "resumes": resumes
        }
        
        start_time = time.time()
        response = requests.post(
            f"{base_url}/ai/rar/v1/analyze_and_rerank", 
            json=data,
            headers=headers
        )
        end_time = time.time()
        
        request_time = end_time - start_time
        
        if response.status_code == 202:
            response_data = response.json()
            trace_id = response_data["trace_id"]
            
            # Save job data
            st.session_state.jobs[trace_id] = {
                "trace_id": trace_id,
                "status": "pending",
                "progress": {},
                "results": None,
                "api_key": api_key,
                "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "last_updated": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "request_time": request_time,
                "status_code": response.status_code
            }
            
            # Update API key usage
            masked_key = get_masked_api_key(api_key)
            if masked_key not in st.session_state.api_key_usage:
                st.session_state.api_key_usage[masked_key] = 0
            st.session_state.api_key_usage[masked_key] += 1
            
            # Update request times
            if masked_key not in st.session_state.request_times:
                st.session_state.request_times[masked_key] = []
            st.session_state.request_times[masked_key].append(request_time)
            
            return trace_id
        else:
            st.error(f"Error submitting job: Status code {response.status_code}")
            return None
    except Exception as e:
        st.error(f"Error submitting job: {str(e)}")
        return None

def refresh_all_jobs(base_url):
    """Refresh status for all jobs"""
    for trace_id, job_data in list(st.session_state.jobs.items()):
        if job_data["status"] not in ["completed", "failed"]:
            update_job_status(trace_id, job_data["api_key"], base_url)

def create_dashboard():
    """Create the main dashboard"""
    # Initialize refresh state
    if "last_refresh" not in st.session_state:
        st.session_state.last_refresh = datetime.now()
    if "refresh_count" not in st.session_state:
        st.session_state.refresh_count = 0

    # Sidebar
    with st.sidebar:
        st.title("API Endpoint Monitor")
        st.markdown("---")
        
        base_url = st.text_input("API Base URL", "http://localhost:8000")
        st.markdown("---")
        
        st.subheader("API Key Management")
        new_api_key = st.text_input("Add New API Key", "")
        key_label = st.text_input("Key Label (optional)", "")
        
        if st.button("Add API Key"):
            if new_api_key:
                masked_key = get_masked_api_key(new_api_key)
                st.session_state.api_keys[masked_key] = new_api_key
                st.success(f"Added API Key: {masked_key}")
        
        st.markdown("---")
        
        st.subheader("Auto Refresh")
        auto_refresh = st.checkbox("Enable Auto-Refresh", value=True)
        refresh_interval = st.slider("Refresh Interval (sec)", 5, 60, 10)
        
        if auto_refresh:
            refresh_placeholder = st.empty()
            # refresh_count = 0
            
            if st.button("Refresh Now"):
                refresh_all_jobs(base_url)
                st.success("Refreshed all jobs!")
                st.session_state.refresh_count = 0
                
            # Auto-refresh logic
            # if "last_refresh" not in st.session_state:
            #     st.session_state.last_refresh = datetime.now() - timedelta(seconds=refresh_interval)
            
            # time_since_refresh = (datetime.now() - st.session_state.last_refresh).total_seconds()
            # Move time check to a more controlled location
            current_time = datetime.now()
            if (current_time - st.session_state.last_refresh).total_seconds() >= refresh_interval:
                refresh_all_jobs(base_url)
                st.session_state.last_refresh = current_time
                st.session_state.refresh_count += 1
                refresh_placeholder.text(f"Last auto-refresh: {current_time.strftime('%H:%M:%S')} (x{st.session_state.refresh_count})")
            # if time_since_refresh >= refresh_interval:
            #     refresh_all_jobs(base_url)
            #     st.session_state.last_refresh = datetime.now()
            #     refresh_count += 1
            #     refresh_placeholder.text(f"Last auto-refresh: {st.session_state.last_refresh.strftime('%H:%M:%S')} (x{refresh_count})")
            # else:
            #     next_refresh = refresh_interval - st.session_state.last_refresh
            #     refresh_placeholder.text(f"Next refresh in: {int(next_refresh)} seconds")
                
    # Main content
    st.title("API Endpoint Monitoring Dashboard")
    
    # Create tabs
    tab1, tab2, tab3, tab4 = st.tabs(["Active Jobs", "Job History", "API Key Analytics", "Submit New Job"])
    
    # Tab 1: Active Jobs
    with tab1:
        st.subheader("Active Jobs")
        
        active_jobs = {trace_id: job for trace_id, job in st.session_state.jobs.items() 
                      if job["status"] in ["pending", "running"]}
        
        if not active_jobs:
            st.info("No active jobs currently")
        else:
            for trace_id, job in active_jobs.items():
                with st.expander(f"Job: {trace_id} - Status: {job['status'].upper()}", expanded=True):
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.markdown(f"**API Key:** <span class='api-key-display'>{get_masked_api_key(job['api_key'])}</span>", unsafe_allow_html=True)
                    with col2:
                        st.markdown(f"**Created:** {job.get('created_at', 'N/A')}")
                    with col3:
                        st.markdown(f"**Last Updated:** {job.get('last_updated', 'N/A')}")
                    
                    # Progress bar
                    progress_placeholder = st.empty()
                    status_class = f"status-{job['status']}"
                    
                    if job["status"] == "pending":
                        progress_placeholder.progress(0)
                        st.markdown(f"<span class='{status_class}'>PENDING</span>: Waiting to start", unsafe_allow_html=True)
                    elif job["status"] == "running":
                        progress_data = job.get("progress", {})
                        total_jobs = len(progress_data)
                        completed_jobs = sum(1 for status in progress_data.values() if status == "completed")
                        
                        if total_jobs > 0:
                            progress_value = completed_jobs / total_jobs
                            progress_placeholder.progress(progress_value)
                            st.markdown(f"<span class='{status_class}'>RUNNING</span>: {completed_jobs}/{total_jobs} tasks completed", unsafe_allow_html=True)
                        else:
                            progress_placeholder.progress(0)
                            st.markdown(f"<span class='{status_class}'>RUNNING</span>: Processing", unsafe_allow_html=True)
                    
                    # Refresh button
                    if st.button(f"Refresh Status (Job: {trace_id[:8]}...)"):
                        if update_job_status(trace_id, job["api_key"], base_url):
                            st.success(f"Status updated for job {trace_id}")
                        else:
                            st.error(f"Failed to update status for job {trace_id}")
    
    # Tab 2: Job History
    with tab2:
        st.subheader("Job History")
        
        # Filters
        col1, col2 = st.columns(2)
        with col1:
            status_filter = st.multiselect(
                "Filter by Status",
                ["completed", "failed", "pending", "running"],
                default=["completed", "failed"]
            )
        with col2:
            if st.session_state.api_keys:
                api_key_filter = st.multiselect(
                    "Filter by API Key",
                    list(st.session_state.api_keys.keys()),
                    default=list(st.session_state.api_keys.keys())
                )
            else:
                api_key_filter = []
                st.warning("No API keys available for filtering")
        
        # Create dataframe from jobs
        if st.session_state.jobs:
            jobs_data = []
            for trace_id, job in st.session_state.jobs.items():
                masked_key = get_masked_api_key(job["api_key"])
                
                if (job["status"] in status_filter) and (not api_key_filter or masked_key in api_key_filter):
                    jobs_data.append({
                        "Trace ID": trace_id,
                        "Status": job["status"].upper(),
                        "API Key": masked_key,
                        "Created": job.get("created_at", "N/A"),
                        "Last Updated": job.get("last_updated", "N/A"),
                        "Request Time (s)": round(job.get("request_time", 0), 2),
                        "Status Code": job.get("status_code", "N/A")
                    })
            
            if jobs_data:
                df = pd.DataFrame(jobs_data)
                
                # Apply styling
                def style_status(val):
                    if val == "COMPLETED":
                        return "background-color: #d4edda; color: #155724"
                    elif val == "FAILED":
                        return "background-color: #f8d7da; color: #721c24"
                    elif val == "RUNNING":
                        return "background-color: #fff3cd; color: #856404"
                    else:  # PENDING
                        return "background-color: #d1ecf1; color: #0c5460"
                
                styled_df = df.style.applymap(style_status, subset=["Status"])
                
                # Sort by creation date (newest first)
                if "Created" in df.columns:
                    df = df.sort_values("Last Updated", ascending=False)
                
                st.dataframe(df, height=400)
                
                # Download button
                csv = df.to_csv(index=False)
                st.download_button(
                    label="Download Job History",
                    data=csv,
                    file_name="job_history.csv",
                    mime="text/csv",
                )
            else:
                st.info("No jobs match the selected filters")
        else:
            st.info("No job history available")
    
    # Tab 3: API Key Analytics
    with tab3:
        st.subheader("API Key Analytics")
        
        if st.session_state.api_key_usage:
            col1, col2 = st.columns(2)
            
            with col1:
                # Usage by API Key
                usage_data = pd.DataFrame({
                    "API Key": list(st.session_state.api_key_usage.keys()),
                    "Requests": list(st.session_state.api_key_usage.values())
                })
                
                fig = px.bar(
                    usage_data,
                    x="API Key",
                    y="Requests",
                    title="API Key Usage",
                    color="Requests",
                    color_continuous_scale="Viridis"
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Average Response Time by API Key
                avg_times = {}
                for key, times in st.session_state.request_times.items():
                    if times:
                        avg_times[key] = sum(times) / len(times)
                
                time_data = pd.DataFrame({
                    "API Key": list(avg_times.keys()),
                    "Avg Response Time (s)": list(avg_times.values())
                })
                
                fig = px.bar(
                    time_data,
                    x="API Key",
                    y="Avg Response Time (s)",
                    title="Average Response Time by API Key",
                    color="Avg Response Time (s)",
                    color_continuous_scale="Reds"
                )
                st.plotly_chart(fig, use_container_width=True)
            
            # Status breakdown
            status_counts = {"completed": 0, "failed": 0, "running": 0, "pending": 0}
            for job in st.session_state.jobs.values():
                if job["status"] in status_counts:
                    status_counts[job["status"]] += 1
            
            status_data = pd.DataFrame({
                "Status": list(status_counts.keys()),
                "Count": list(status_counts.values())
            })
            
            fig = px.pie(
                status_data,
                values="Count",
                names="Status",
                title="Job Status Distribution",
                color="Status",
                color_discrete_map={
                    "completed": "green",
                    "failed": "red",
                    "running": "orange",
                    "pending": "blue"
                }
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Response Time Distribution
            all_times = []
            for times in st.session_state.request_times.values():
                all_times.extend(times)
            
            if all_times:
                fig = px.histogram(
                    x=all_times,
                    nbins=20,
                    title="Response Time Distribution (seconds)",
                    labels={"x": "Response Time (s)"},
                    color_discrete_sequence=["lightblue"]
                )
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No API key usage data available yet")
    
    # Tab 4: Submit New Job
    with tab4:
        st.subheader("Submit New Job")
        
        if not st.session_state.api_keys:
            st.warning("No API keys available. Please add an API key in the sidebar first.")
        else:
            selected_api_key_masked = st.selectbox(
                "Select API Key",
                options=list(st.session_state.api_keys.keys())
            )
            
            selected_api_key = st.session_state.api_keys.get(selected_api_key_masked)
            
            # Job data input
            st.subheader("Job Openings")
            job_openings_json = st.text_area(
                "Job Openings JSON (List of dictionaries)",
                value='[{"name": "Software Engineer", "description": "Python dev"}]',
                height=150
            )
            
            st.subheader("Resumes")
            resumes_json = st.text_area(
                "Resumes JSON (List of dictionaries)",
                value='[{"page_content": "Experienced Python developer", "metadata": {"name": "John Doe"}}]',
                height=150
            )
            
            if st.button("Submit Job"):
                try:
                    job_openings = json.loads(job_openings_json)
                    resumes = json.loads(resumes_json)
                    
                    if not isinstance(job_openings, list) or not isinstance(resumes, list):
                        st.error("Both job openings and resumes must be lists")
                    else:
                        trace_id = submit_job(job_openings, resumes, selected_api_key, base_url)
                        
                        if trace_id:
                            st.success(f"Job submitted successfully! Trace ID: {trace_id}")
                            st.info("Switch to the 'Active Jobs' tab to monitor progress")
                except json.JSONDecodeError:
                    st.error("Invalid JSON format. Please check your input")
                except Exception as e:
                    st.error(f"Error submitting job: {str(e)}")

# Run the app
if __name__ == "__main__":
    create_dashboard()