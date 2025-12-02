import tkinter as tk
from tkinter import filedialog, ttk, messagebox
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from tkinter import simpledialog
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans


# Optional improved theming
try:
    import ttkbootstrap as tb
except Exception:
    tb = None
try:
    import ner as ner_module
except Exception:
    ner_module = None


def normalize_columns(df: pd.DataFrame, for_prediction: bool = False) -> pd.DataFrame:
    """Normalize common column name variants to standard names."""
    col_map = {}
    lower_cols = {c.lower(): c for c in df.columns}

    def find_variant(possible):
        for p in possible:
            if p.lower() in lower_cols:
                return lower_cols[p.lower()]
        for col in df.columns:
            lower_col = col.lower()
            for p in possible:
                if p.lower() in lower_col:
                    return col
        return None

    # common variants
    status_col = find_variant(['Status', 'studentstatus', 'student_status', 'enrollment_status', 'state'])
    hasjob_col = find_variant(['hasjob', 'HasSideJob', 'has_side_jobs', 'hasjobs', 'hassidejobs', 'employed', 'HasSideJobs'])
    income_col = find_variant(['MonthlyFamilyIncome', 'income', 'pay', 'wage', 'compensation'])

    if status_col:
        col_map[status_col] = 'Status'
    if hasjob_col:
        col_map[hasjob_col] = 'HasSideJob'
    if income_col:
        col_map[income_col] = 'MonthlyFamilyIncome'

    if col_map:
        df = df.rename(columns=col_map)

    # Only normalize Status if NOT doing prediction (to avoid data leakage)
    if not for_prediction and 'Status' in df.columns:
        if pd.api.types.is_numeric_dtype(df['Status']):
            df['Status'] = df['Status'].fillna(0).astype(int).clip(0, 1)
        else:
            df['Status'] = df['Status'].astype(str).str.lower().str.strip().apply(
                lambda x: 1 if 'enrolled' in x else 0
            ).astype(int)

    # Normalize HasSideJob
    if 'HasSideJob' in df.columns or 'hasjob' in df.columns:
        try:
            if 'hasjob' in df.columns and 'HasSideJob' not in df.columns:
                df = df.rename(columns={'hasjob': 'HasSideJob'})
            
            df['HasSideJob'] = df['HasSideJob'].map(
                lambda v: 1 if str(v).strip().lower() in ('1', 'true', 'yes', 'y', 't')
                else (0 if str(v).strip().lower() in ('0', 'false', 'no', 'n', 'f') else v)
            )
            df['HasSideJob'] = pd.to_numeric(df['HasSideJob'], errors='coerce').fillna(0).astype(int)
        except Exception:
            pass

    # Normalize MonthlyFamilyIncome
    if 'MonthlyFamilyIncome' in df.columns:
        # For prediction, we need to handle this differently to avoid data leakage
        if for_prediction:
            # Use the same encoding as training - you might need to pass a pre-fitted encoder
            pass
        else:
            # Convert categorical income to numeric codes
            if not pd.api.types.is_numeric_dtype(df['MonthlyFamilyIncome']):
                df['MonthlyFamilyIncome'] = df['MonthlyFamilyIncome'].astype('category').cat.codes
            # Scale to 0-1 range for better model performance
            df['MonthlyFamilyIncome'] = (df['MonthlyFamilyIncome'] - df['MonthlyFamilyIncome'].min()) / (df['MonthlyFamilyIncome'].max() - df['MonthlyFamilyIncome'].min())

    return df


# Replace the cluster_students and cluster_faculty functions with these safer versions:

def cluster_students():
    """Cluster students using Age + GWA + Status + Year Level"""
    try:
        if 'df' not in globals() or df is None or df.empty:
            messagebox.showerror("Error", "No data available for clustering.")
            return

        # Normalize column names
        data = df.copy()
        
        # Find relevant columns using common variants
        def find_column(variants):
            for variant in variants:
                for col in data.columns:
                    if variant.lower() in col.lower():
                        return col
            return None

        age_col = find_column(['Age', 'age'])
        gwa_col = find_column(['GWA', 'gwa', 'Grade', 'grade', 'Score', 'score'])
        status_col = find_column(['Status', 'status', 'StudentStatus'])
        year_col = find_column(['Year', 'year', 'YearLevel', 'yearlevel', 'Level', 'level'])

        required_cols = [age_col, gwa_col, status_col, year_col]
        missing_cols = [f"'{variants[0]}'" for col, variants in 
                       zip(required_cols, [['Age'], ['GWA'], ['Status'], ['Year']]) 
                       if col is None]
        
        if missing_cols:
            messagebox.showerror("Error", f"Missing required columns: {', '.join(missing_cols)}")
            return

        # Prepare data for clustering
        cluster_data = data[[age_col, gwa_col, status_col, year_col]].copy()
        
        # Convert categorical columns to numeric if needed
        if not pd.api.types.is_numeric_dtype(cluster_data[status_col]):
            cluster_data[status_col] = cluster_data[status_col].astype('category').cat.codes
        
        if not pd.api.types.is_numeric_dtype(cluster_data[year_col]):
            cluster_data[year_col] = cluster_data[year_col].astype('category').cat.codes

        # Handle missing values
        if cluster_data.isnull().any().any():
            cluster_data = cluster_data.fillna(cluster_data.mean())
            messagebox.showinfo("Info", "Missing values filled with column means")

        # Scale the data
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(cluster_data)

        # Determine optimal number of clusters using elbow method
        inertias = []
        k_range = range(2, 8)
        
        for k in k_range:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            kmeans.fit(scaled_data)
            inertias.append(kmeans.inertia_)

        # Use elbow method to find optimal k (simplified - use 3 clusters)
        optimal_k = 3
        
        # Perform clustering
        kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
        clusters = kmeans.fit_predict(scaled_data)
        
        # Add cluster labels to data
        data['Student_Cluster'] = clusters
        cluster_data['Cluster'] = clusters

        # Create visualization
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        fig.patch.set_facecolor("#1E1E2F")
        
        # Plot 1: Age vs GWA colored by cluster
        scatter1 = ax1.scatter(data[age_col], data[gwa_col], c=clusters, cmap='viridis', alpha=0.7)
        ax1.set_xlabel(age_col, color="#F8F8F2")
        ax1.set_ylabel(gwa_col, color="#F8F8F2")
        ax1.set_title(f"Student Clusters: {age_col} vs {gwa_col}", color="#F8F8F2")
        ax1.tick_params(colors="#F8F8F2")
        ax1.grid(True, alpha=0.3)
        plt.colorbar(scatter1, ax=ax1)

        # Plot 2: Cluster distribution
        cluster_counts = data['Student_Cluster'].value_counts().sort_index()
        bars = ax2.bar(cluster_counts.index, cluster_counts.values, color=['#FF6B6B', '#4ECDC4', '#45B7D1'])
        ax2.set_xlabel('Cluster', color="#F8F8F2")
        ax2.set_ylabel('Number of Students', color="#F8F8F2")
        ax2.set_title('Student Distribution Across Clusters', color="#F8F8F2")
        ax2.tick_params(colors="#F8F8F2")
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(height)}', ha='center', va='bottom', color="#F8F8F2")

        # Set background colors
        for ax in [ax1, ax2]:
            ax.set_facecolor("#1E1E2F")

        plt.tight_layout()

        # Safe UI updates
        def safe_ui_updates():
            try:
                # Display in visualization frame
                for widget in visualization_frame.winfo_children():
                    widget.destroy()
                canvas = FigureCanvasTkAgg(fig, master=visualization_frame)
                canvas.draw()
                canvas.get_tk_widget().pack(fill="both", expand=True)

                # Show cluster summary
                cluster_summary = data.groupby('Student_Cluster').agg({
                    age_col: 'mean',
                    gwa_col: 'mean',
                    status_col: lambda x: x.mode()[0] if len(x.mode()) > 0 else 'Unknown',
                    year_col: 'mean'
                }).round(2)

                summary_text = f"Student Clustering Results (K={optimal_k} clusters)\n\n"
                summary_text += f"Features used: {age_col}, {gwa_col}, {status_col}, {year_col}\n\n"
                summary_text += "Cluster Profiles:\n"
                
                for cluster_id in range(optimal_k):
                    cluster_data_subset = data[data['Student_Cluster'] == cluster_id]
                    summary_text += f"\nCluster {cluster_id} ({len(cluster_data_subset)} students):\n"
                    summary_text += f"  Avg {age_col}: {cluster_data_subset[age_col].mean():.1f}\n"
                    summary_text += f"  Avg {gwa_col}: {cluster_data_subset[gwa_col].mean():.2f}\n"
                    summary_text += f"  Avg {year_col}: {cluster_data_subset[year_col].mean():.1f}\n"

                # Update relationship label safely
                if relationship_label.winfo_exists():
                    relationship_label.config(text=f"Student Clustering Complete - {optimal_k} clusters identified")

                # Show detailed results in auxiliary panel safely
                if str(aux_panel) not in left_vpaned.panes():
                    left_vpaned.add(aux_panel)
                aux_var.set(True)

                # Clear auxiliary panel safely
                for widget in aux_panel.winfo_children():
                    if widget != aux_header and widget.winfo_exists():
                        widget.destroy()

                text_frame = tk.Frame(aux_panel, bg="#1E1E2F")
                text_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=(0,10))

                results_text = tk.Text(text_frame, wrap=tk.WORD, bg="#282A36", fg="#F8F8F2", font=("Arial", 10))
                results_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

                scrollbar = ttk.Scrollbar(text_frame, orient=tk.VERTICAL, command=results_text.yview)
                scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
                results_text.configure(yscrollcommand=scrollbar.set)

                results_text.insert(tk.END, summary_text)
                results_text.config(state=tk.DISABLED)

                messagebox.showinfo("Success", f"Student clustering completed! {optimal_k} clusters identified.")
                
            except Exception as ui_error:
                print(f"UI update error: {ui_error}")
                # Fallback: just show a simple message
                try:
                    messagebox.showinfo("Success", f"Student clustering completed with {optimal_k} clusters!")
                except:
                    pass

        # Schedule UI updates to run in the main thread
        root.after(0, safe_ui_updates)

    except Exception as e:
        # Handle the error in the main thread
        def show_error():
            messagebox.showerror("Clustering Error", f"An error occurred: {e}")
        root.after(0, show_error)

def cluster_faculty():
    """Cluster faculty using Experience + Teaching Load + Age"""
    try:
        if 'df' not in globals() or df is None or df.empty:
            messagebox.showerror("Error", "No data available for clustering.")
            return

        # Normalize column names
        data = df.copy()
        
        # Find relevant columns using common variants
        def find_column(variants):
            for variant in variants:
                for col in data.columns:
                    if variant.lower() in col.lower():
                        return col
            return None

        exp_col = find_column(['Experience', 'experience', 'Exp', 'exp', 'Years', 'years'])
        load_col = find_column(['TeachingLoad', 'teachingload', 'Load', 'load', 'Teaching', 'teaching'])
        age_col = find_column(['Age', 'age'])

        required_cols = [exp_col, load_col, age_col]
        missing_cols = [f"'{variants[0]}'" for col, variants in 
                       zip(required_cols, [['Experience'], ['TeachingLoad'], ['Age']]) 
                       if col is None]
        
        if missing_cols:
            messagebox.showerror("Error", f"Missing required columns: {', '.join(missing_cols)}")
            return

        # Prepare data for clustering
        cluster_data = data[[exp_col, load_col, age_col]].copy()

        # Handle missing values
        if cluster_data.isnull().any().any():
            cluster_data = cluster_data.fillna(cluster_data.mean())
            messagebox.showinfo("Info", "Missing values filled with column means")

        # Scale the data
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(cluster_data)

        # Use 3 clusters for faculty (low, medium, high load)
        optimal_k = 3
        
        # Perform clustering
        kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
        clusters = kmeans.fit_predict(scaled_data)
        
        # Add cluster labels to data with meaningful names
        cluster_names = {0: 'Low Load', 1: 'Medium Load', 2: 'High Load'}
        data['Faculty_Cluster'] = [cluster_names.get(cluster, f'Cluster {cluster}') for cluster in clusters]

        # Create visualization
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        fig.patch.set_facecolor("#1E1E2F")
        
        # Plot 1: Experience vs Teaching Load colored by cluster
        scatter1 = ax1.scatter(data[exp_col], data[load_col], c=clusters, cmap='plasma', alpha=0.7)
        ax1.set_xlabel(exp_col, color="#F8F8F2")
        ax1.set_ylabel(load_col, color="#F8F8F2")
        ax1.set_title(f"Faculty Clusters: {exp_col} vs {load_col}", color="#F8F8F2")
        ax1.tick_params(colors="#F8F8F2")
        ax1.grid(True, alpha=0.3)
        plt.colorbar(scatter1, ax=ax1)

        # Plot 2: Cluster distribution
        cluster_counts = data['Faculty_Cluster'].value_counts()
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1'][:len(cluster_counts)]
        bars = ax2.bar(range(len(cluster_counts)), cluster_counts.values, color=colors)
        ax2.set_xlabel('Faculty Group', color="#F8F8F2")
        ax2.set_ylabel('Number of Faculty', color="#F8F8F2")
        ax2.set_title('Faculty Distribution Across Groups', color="#F8F8F2")
        ax2.tick_params(colors="#F8F8F2")
        ax2.set_xticks(range(len(cluster_counts)))
        ax2.set_xticklabels(cluster_counts.index, rotation=45)
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(height)}', ha='center', va='bottom', color="#F8F8F2")

        # Set background colors
        for ax in [ax1, ax2]:
            ax.set_facecolor("#1E1E2F")

        plt.tight_layout()

        # Safe UI updates
        def safe_ui_updates():
            try:
                # Display in visualization frame
                for widget in visualization_frame.winfo_children():
                    if widget.winfo_exists():
                        widget.destroy()
                canvas = FigureCanvasTkAgg(fig, master=visualization_frame)
                canvas.draw()
                canvas.get_tk_widget().pack(fill="both", expand=True)

                # Show cluster summary
                summary_text = f"Faculty Clustering Results\n\n"
                summary_text += f"Features used: {exp_col}, {load_col}, {age_col}\n\n"
                summary_text += "Faculty Groups:\n"
                
                for cluster_name in cluster_names.values():
                    cluster_data_subset = data[data['Faculty_Cluster'] == cluster_name]
                    if len(cluster_data_subset) > 0:
                        summary_text += f"\n{cluster_name} Group ({len(cluster_data_subset)} faculty):\n"
                        summary_text += f"  Avg {exp_col}: {cluster_data_subset[exp_col].mean():.1f} years\n"
                        summary_text += f"  Avg {load_col}: {cluster_data_subset[load_col].mean():.1f}\n"
                        summary_text += f"  Avg {age_col}: {cluster_data_subset[age_col].mean():.1f} years\n"

                # Update relationship label safely
                if relationship_label.winfo_exists():
                    relationship_label.config(text=f"Faculty Clustering Complete - {optimal_k} groups identified")

                # Show detailed results in auxiliary panel safely
                if str(aux_panel) not in left_vpaned.panes():
                    left_vpaned.add(aux_panel)
                aux_var.set(True)

                # Clear auxiliary panel safely
                for widget in aux_panel.winfo_children():
                    if widget != aux_header and widget.winfo_exists():
                        widget.destroy()

                text_frame = tk.Frame(aux_panel, bg="#1E1E2F")
                text_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=(0,10))

                results_text = tk.Text(text_frame, wrap=tk.WORD, bg="#282A36", fg="#F8F8F2", font=("Arial", 10))
                results_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

                scrollbar = ttk.Scrollbar(text_frame, orient=tk.VERTICAL, command=results_text.yview)
                scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
                results_text.configure(yscrollcommand=scrollbar.set)

                results_text.insert(tk.END, summary_text)
                results_text.config(state=tk.DISABLED)

                messagebox.showinfo("Success", f"Faculty clustering completed! {optimal_k} faculty groups identified.")
                
            except Exception as ui_error:
                print(f"UI update error: {ui_error}")
                # Fallback: just show a simple message
                try:
                    messagebox.showinfo("Success", f"Faculty clustering completed with {optimal_k} groups!")
                except:
                    pass

        # Schedule UI updates to run in the main thread
        root.after(0, safe_ui_updates)

    except Exception as e:
        # Handle the error in the main thread
        def show_error():
            messagebox.showerror("Clustering Error", f"An error occurred: {e}")
        root.after(0, show_error)

# Function to upload CSV file
def upload_file():
    global df
    file_path = filedialog.askopenfilename(title="Select a CSV file", filetypes=[("CSV files", "*.csv")])
    if file_path:
        try:
            df = pd.read_csv(file_path)
            # Populate Data Table
            data_tree["columns"] = df.columns.tolist()  # Set columns dynamically
            for col in df.columns:
                data_tree.heading(col, text=col)
                data_tree.column(col, width=150, anchor="center")

            for row in df.itertuples(index=False):
                data_tree.insert("", "end", values=row)  # Insert rows into Treeview

            # Enable Graph Customization
            columns = df.columns.tolist()
            column_var.set(columns[0])  # Default to the first column
            # Default second column to the second available column if present
            column2_var.set(columns[1] if len(columns) > 1 else columns[0])
            column_menu["menu"].delete(0, "end")
            column2_menu["menu"].delete(0, "end")
            for column in columns:
                column_menu["menu"].add_command(label=column, command=tk._setit(column_var, column))
                column2_menu["menu"].add_command(label=column, command=tk._setit(column2_var, column))
            
            
            # Add sorting columns to the new sort_menu including "Don't sort" option as first
            sort_column_var.set("Don't sort")
            sort_menu["menu"].delete(0, "end")
            sort_menu["menu"].add_command(label="Don't sort", command=tk._setit(sort_column_var, "Don't sort"))
            for column in columns:
                sort_menu["menu"].add_command(label=column, command=tk._setit(sort_column_var, column))
            
            # Enable other buttons after file is uploaded
            visualize_button.config(state=tk.NORMAL)
            analyze_button.config(state=tk.NORMAL)
            save_button.config(state=tk.NORMAL)
            # Enable Recommend Faculty button when data is loaded
            try:
                recommend_button.config(state=tk.NORMAL)
            except NameError:
                pass
        except Exception as e:
            messagebox.showerror("Error", f"An error occurred: {e}")

def visualize_data():
    try:
        column = column_var.get()
        column2 = column2_var.get()
        graph_type = graph_type_var.get()
        color = color_var.get()

        # Check if the selected column is valid
        if column not in df.columns:
            messagebox.showerror("Error", f"Column '{column}' not found!")
            return

        # Work on a copy to avoid modifying original data
        temp_df = df.copy()

        # Normalize columns for visualization (don't modify global df)
        temp_df = normalize_columns(temp_df, for_prediction=False)

        # Get the data for the selected column
        data_column = temp_df[column]

        # For scatter and line plots, use column2 as x-axis when provided
        if graph_type in ("Scatter", "Line"):
            if column2 not in temp_df.columns:
                messagebox.showerror("Error", f"Second column '{column2}' not found!")
                return
            x_data = temp_df[column2]
            y_data = data_column
        else:
            x_data = None
            y_data = data_column

        # Attempt to coerce categorical columns to numeric codes when appropriate
        def coerce_if_categorical(series, name):
            # If already numeric, return as-is
            if pd.api.types.is_numeric_dtype(series):
                return series
            # For object / categorical types, convert to category codes
            if series.dtype == 'object' or pd.api.types.is_categorical_dtype(series):
                try:
                    return series.astype('category').cat.codes
                except Exception:
                    pass
            # Otherwise return original
            return series

        # For plots that require numeric y-values (Histogram, Line, Scatter), coerce if needed
        if graph_type in ("Histogram", "Line", "Scatter"):
            y_data = coerce_if_categorical(y_data, column)
            if not pd.api.types.is_numeric_dtype(y_data):
                messagebox.showerror("Error", f"Column '{column}' must be numeric for this graph type.")
                return

        # For scatter and line plots, also ensure x is numeric (coerce categorical to codes)
        if graph_type in ("Scatter", "Line") and x_data is not None:
            x_data = coerce_if_categorical(x_data, column2)
            if not pd.api.types.is_numeric_dtype(x_data):
                messagebox.showerror("Error", f"Column '{column2}' must be numeric for this plot type.")
                return

        # Create a new figure for visualization
        fig, ax = plt.subplots(figsize=(8, 6))

        # Set figure and axes background color
        fig.patch.set_facecolor("#1E1E2F")  # Matches the app background
        ax.set_facecolor("#1E1E2F")  # Matches the app background

        # Visualization logic based on the selected graph type
        if graph_type == "Histogram":
            # If second column selected and different, try to show distribution by the second column
            if column2 and column2 in temp_df.columns and column2 != column:
                # If column2 is categorical, use it as hue to split the histogram
                if temp_df[column2].dtype == 'object' or pd.api.types.is_categorical_dtype(temp_df[column2]):
                    try:
                        sns.histplot(data=temp_df, x=column, hue=column2, multiple='stack', ax=ax)
                        ax.set_title(f"Histogram of {column} by {column2}")
                        ax.set_xlabel(column)
                    except Exception:
                        # Fallback to single histogram
                        sns.histplot(y_data, kde=True, color=color, ax=ax)
                        ax.set_title(f"Histogram of {column}")
                        ax.set_xlabel(column)
                else:
                    # If both numeric, overlay two histograms for comparison
                    try:
                        sns.histplot(temp_df[column].dropna(), kde=True, color=color, ax=ax, stat='density', alpha=0.5, label=column)
                        other_color = 'orange' if color != 'orange' else 'green'
                        sns.histplot(temp_df[column2].dropna(), kde=True, color=other_color, ax=ax, stat='density', alpha=0.5, label=column2)
                        ax.legend()
                        ax.set_title(f"Overlayed Histograms: {column} and {column2}")
                        ax.set_xlabel('Value')
                    except Exception:
                        sns.histplot(y_data, kde=True, color=color, ax=ax)
                        ax.set_title(f"Histogram of {column}")
                        ax.set_xlabel(column)
            else:
                sns.histplot(y_data, kde=True, color=color, ax=ax)
                ax.set_title(f"Histogram of {column}")
                ax.set_xlabel(column)

        elif graph_type == "Bar":
            # If second column provided, show grouped/count or aggregated bars
            if column2 and column2 in temp_df.columns and column2 != column:
                # If both columns categorical -> grouped counts
                if (temp_df[column].dtype == 'object' or pd.api.types.is_categorical_dtype(temp_df[column])) and \
                   (temp_df[column2].dtype == 'object' or pd.api.types.is_categorical_dtype(temp_df[column2])):
                    try:
                        sns.countplot(x=column, hue=column2, data=temp_df, ax=ax)
                        ax.set_title(f"Grouped counts of {column} by {column2}")
                        ax.set_xlabel(column)
                        ax.set_ylabel('Count')
                    except Exception:
                        value_counts = y_data.value_counts()
                        sns.barplot(x=value_counts.index, y=value_counts.values, color=color, ax=ax)
                        ax.set_title(f"Bar Graph of {column}")
                        ax.set_xlabel(column)
                        ax.set_ylabel("Count")
                # If column numeric and column2 categorical -> aggregated (mean) bars per category
                elif pd.api.types.is_numeric_dtype(temp_df[column]) and (temp_df[column2].dtype == 'object' or pd.api.types.is_categorical_dtype(temp_df[column2])):
                    try:
                        grouped = temp_df.groupby(column2)[column].mean().sort_values()
                        sns.barplot(x=grouped.index.astype(str), y=grouped.values, color=color, ax=ax)
                        ax.set_title(f"Average {column} by {column2}")
                        ax.set_xlabel(column2)
                        ax.set_ylabel(f"Avg {column}")
                    except Exception:
                        value_counts = y_data.value_counts()
                        sns.barplot(x=value_counts.index, y=value_counts.values, color=color, ax=ax)
                        ax.set_title(f"Bar Graph of {column}")
                        ax.set_xlabel(column)
                        ax.set_ylabel("Count")
                else:
                    # Fallback: grouped counts using crosstab and plot
                    try:
                        ct = pd.crosstab(temp_df[column2].astype(str), temp_df[column].astype(str))
                        ct.plot(kind='bar', ax=ax)
                        ax.set_title(f"Counts of {column} grouped by {column2}")
                        ax.set_xlabel(column2)
                        ax.set_ylabel('Count')
                    except Exception:
                        value_counts = y_data.value_counts()
                        sns.barplot(x=value_counts.index, y=value_counts.values, color=color, ax=ax)
                        ax.set_title(f"Bar Graph of {column}")
                        ax.set_xlabel(column)
                        ax.set_ylabel("Count")
            else:
                # For bar plots, count unique values (single-column)
                value_counts = y_data.value_counts()
                sns.barplot(x=value_counts.index, y=value_counts.values, color=color, ax=ax)
                ax.set_title(f"Bar Graph of {column}")
                ax.set_xlabel(column)
                ax.set_ylabel("Count")

        elif graph_type == "Scatter":
            sns.scatterplot(x=x_data, y=y_data, color=color, ax=ax)
            ax.set_title(f"Scatter Plot of {column} vs {column2}")
            ax.set_xlabel(column2)
            ax.set_ylabel(column)

        elif graph_type == "Line":
            # For line plots, if a second column was provided use it as x; otherwise use index
            try:
                if x_data is not None:
                    # Build a small DataFrame for plotting and drop NA
                    df_line = temp_df[[column2, column]].copy().dropna()

                    # Coerce categorical columns to numeric codes when needed
                    df_line[column2] = coerce_if_categorical(df_line[column2], column2)
                    df_line[column] = coerce_if_categorical(df_line[column], column)

                    # If x has duplicate values, aggregate by mean to produce a smooth, meaningful line
                    if df_line[column2].duplicated().any():
                        df_line = df_line.groupby(column2, as_index=False)[column].mean()
                        try:
                            if relationship_label.winfo_exists():
                                relationship_label.config(text=f"Line plot aggregated mean {column} per {column2}")
                        except Exception:
                            pass
                        # Inform the user that aggregation was applied
                        try:
                            messagebox.showinfo("Aggregation Applied", f"Multiple identical {column2} values found.\nDisplayed line shows mean {column} per {column2}.")
                        except Exception:
                            pass

                    # Sort by x to avoid zig-zag lines
                    df_line = df_line.sort_values(by=column2)

                    ax.plot(df_line[column2].values, df_line[column].values, color=color, marker='o')
                    ax.set_title(f"Line Graph of {column} vs {column2}")
                    ax.set_xlabel(column2)
                else:
                    ax.plot(range(len(y_data)), y_data.values, color=color, marker='o')
                    ax.set_title(f"Line Graph of {column}")
                    ax.set_xlabel("Index")
                ax.set_ylabel(column)
            except Exception:
                # Fallback to index-based line
                ax.plot(range(len(y_data)), y_data.values, color=color, marker='o')
                ax.set_title(f"Line Graph of {column}")
                ax.set_xlabel("Index")
                ax.set_ylabel(column)

        # Adjust label and title colors to match the theme
        ax.title.set_color("#F8F8F2")  # Title color
        ax.xaxis.label.set_color("#F8F8F2")  # X-axis label color
        ax.yaxis.label.set_color("#F8F8F2")  # Y-axis label color
        ax.tick_params(colors="#F8F8F2")  # Tick label color

        # Clear previous visualizations and display the new one
        for widget in visualization_frame.winfo_children():
            widget.destroy()
        canvas = FigureCanvasTkAgg(fig, master=visualization_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True)

    except Exception as e:
        messagebox.showerror("Error", f"An error occurred while visualizing: {e}")



# Function to save the visualized data
def save_visualization():
    try:
        # Ask the user for the file path to save the image
        file_path = filedialog.asksaveasfilename(
            title="Save Visualization",
            defaultextension=".png",  # Default file extension
            filetypes=[("PNG files", "*.png"), ("JPEG files", "*.jpg"), ("All files", "*.*")]
        )
        if file_path:
            # Save the current figure
            plt.savefig(file_path)
            messagebox.showinfo("Success", f"Visualization saved as: {file_path}")
    except Exception as e:
        messagebox.showerror("Error", f"An error occurred while saving: {e}")


def analyze_relationship():
    try:
        col1 = column_var.get()
        col2 = column2_var.get()
        graph_type = graph_type_var.get()

        if col1 not in df.columns or col2 not in df.columns:
            messagebox.showerror("Error", "Please select valid columns.")
            return

        # Create temporary columns for analysis without modifying the original df
        temp_col1 = df[col1]
        if temp_col1.dtype == 'object':
            temp_col1 = temp_col1.astype("category").cat.codes
            messagebox.showinfo("Conversion", f"Column '{col1}' converted to numeric codes temporarily for analysis.")

        temp_col2 = df[col2]
        if temp_col2.dtype == 'object':  # If it's a string/categorical column
            temp_col2 = temp_col2.astype("category").cat.codes
            messagebox.showinfo("Conversion", f"Column '{col2}' converted to numeric codes temporarily for analysis.")

        # Check if the temporary columns are numeric
        if not pd.api.types.is_numeric_dtype(temp_col1) or not pd.api.types.is_numeric_dtype(temp_col2):
            messagebox.showerror("Error", "Both columns must be numeric for this analysis.")
            return

        # Correlation and visualization logic
        correlation = temp_col1.corr(temp_col2)
        relationship_label.config(text=f"Correlation between {col1} and {col2}: {correlation:.2f}")

        # Create figure and axis
        fig, ax = plt.subplots(figsize=(8, 6))

        # Set figure and axes background color to match the base background color
        fig.patch.set_facecolor("#1E1E2F")  # Matches the app background
        ax.set_facecolor("#1E1E2F")  # Matches the app background

        sns.scatterplot(x=temp_col1, y=temp_col2, color=color_var.get(), ax=ax)
        ax.set_title(f"Scatter Plot of {col1} vs {col2}")
        ax.set_xlabel(col1)
        ax.set_ylabel(col2)

        # Adjust label and title colors to match the theme
        ax.title.set_color("#F8F8F2")  # Title color
        ax.xaxis.label.set_color("#F8F8F2")  # X-axis label color
        ax.yaxis.label.set_color("#F8F8F2")  # Y-axis label color
        ax.tick_params(colors="#F8F8F2")  # Tick label color

        # Clear previous visualizations and display the new one
        for widget in visualization_frame.winfo_children():
            widget.destroy()
        canvas = FigureCanvasTkAgg(fig, master=visualization_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True)

    except Exception as e:
        messagebox.showerror("Error", f"An error occurred: {e}")

def Predictions():
    """Run a simple supervised prediction using the normalized category columns."""
    try:
        if 'df' not in globals() or df is None or df.empty:
            messagebox.showerror("Error", "No data available for predictions.")
            return

        # First, let's check the original data size
        original_size = len(df)
        messagebox.showinfo("Data Info", f"Original dataset size: {original_size} students")

        # Work on a copy of the original data
        data = df.copy()
        
        # Manual normalization WITHOUT dropping any rows yet
        if 'Status' in data.columns:
            if pd.api.types.is_numeric_dtype(data['Status']):
                data['Status'] = data['Status'].fillna(0).astype(int).clip(0, 1)
            else:
                data['Status'] = data['Status'].astype(str).str.lower().str.strip().apply(
                    lambda x: 1 if 'enrolled' in x else 0
                ).astype(int)

        # Handle HasSideJob column
        if 'HasSideJob' in data.columns:
            data['HasSideJob'] = data['HasSideJob'].map(
                lambda v: 1 if str(v).strip().lower() in ('1', 'true', 'yes', 'y', 't')
                else (0 if str(v).strip().lower() in ('0', 'false', 'no', 'n', 'f') else v)
            )
            data['HasSideJob'] = pd.to_numeric(data['HasSideJob'], errors='coerce')

        # Handle MonthlyFamilyIncome
        if 'MonthlyFamilyIncome' in data.columns:
            if not pd.api.types.is_numeric_dtype(data['MonthlyFamilyIncome']):
                data['MonthlyFamilyIncome'] = data['MonthlyFamilyIncome'].astype('category').cat.codes
            # Scale to 0-1 range
            data['MonthlyFamilyIncome'] = (data['MonthlyFamilyIncome'] - data['MonthlyFamilyIncome'].min()) / (data['MonthlyFamilyIncome'].max() - data['MonthlyFamilyIncome'].min())

        # Check if we have the required columns
        required = ['HasSideJob', 'MonthlyFamilyIncome', 'Status']
        missing = [c for c in required if c not in data.columns]
        if missing:
            messagebox.showerror("Error", f"Missing required columns: {', '.join(missing)}")
            return

        # Check for missing values BEFORE splitting
        missing_info = data[required].isnull().sum()
        if missing_info.sum() > 0:
            messagebox.showwarning("Missing Data", 
                                 f"Missing values found:\n{missing_info}\n\nThese rows will be removed.")
        
        # Remove rows with missing values in required columns
        data_clean = data[required].dropna()
        rows_removed = len(data) - len(data_clean)
        
        if rows_removed > 0:
            messagebox.showinfo("Data Cleaning", f"Removed {rows_removed} rows with missing values")
        
        if len(data_clean) == 0:
            messagebox.showerror("Error", "No data remaining after cleaning!")
            return

        # Prepare features and target
        X = data_clean[['HasSideJob', 'MonthlyFamilyIncome']].copy()
        y = data_clean['Status'].copy()

        # Check class distribution
        enrolled_count = y.sum()
        dropped_count = len(y) - enrolled_count
        
        messagebox.showinfo("Class Distribution", 
                          f"Total students: {len(y)}\n"
                          f"Enrolled: {enrolled_count} ({enrolled_count/len(y)*100:.1f}%)\n"
                          f"Dropped: {dropped_count} ({dropped_count/len(y)*100:.1f}%)")

        if enrolled_count == 0 or dropped_count == 0:
            messagebox.showerror("Error", f"Need both enrolled and dropped students for prediction.")
            return

        # Import sklearn and other required libraries
        try:
            from sklearn.model_selection import train_test_split, cross_val_score
            from sklearn.ensemble import RandomForestClassifier
            from sklearn.preprocessing import StandardScaler
            from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
            from sklearn.utils.class_weight import compute_class_weight
            import numpy as np
            import seaborn as sns  # Add this import
            import matplotlib.colors as mcolors  # Add this import
        except Exception as e:
            messagebox.showerror("Missing package", f"Please install required packages: {e}")
            return

        # Split data - use 80% train, 20% test
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Compute class weights to handle imbalance (with safety check)
        try:
            class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
            class_weight_dict = {0: class_weights[0], 1: class_weights[1]}
        except Exception:
            # Fallback if class_weight computation fails
            class_weight_dict = 'balanced'

        # Train model with class weights
        model = RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            class_weight=class_weight_dict,
            max_depth=5
        )
        
        model.fit(X_train_scaled, y_train)

        # Cross-validation
        cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='accuracy')
        
        # Predictions
        y_pred = model.predict(X_test_scaled)
        X_full_scaled = scaler.transform(X)
        y_pred_full = model.predict(X_full_scaled)

        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        full_accuracy = accuracy_score(y, y_pred_full)

        # Show comprehensive results
        result_msg = f"""
Data Processing:
- Original dataset: {original_size} students
- After cleaning: {len(data_clean)} students
- Rows removed: {rows_removed}

Training/Test Split:
- Training data: {len(X_train)} students
- Test data: {len(X_test)} students

Class Distribution (Training):
- Enrolled: {y_train.sum()} students ({y_train.sum()/len(y_train)*100:.1f}%)
- Dropped: {len(y_train) - y_train.sum()} students ({(len(y_train) - y_train.sum())/len(y_train)*100:.1f}%)

Prediction Results:
- Test Accuracy: {accuracy:.3f}
- Cross-validation Scores: {[f'{s:.3f}' for s in cv_scores]}
- Full Data Accuracy: {full_accuracy:.3f}

Feature Importance:
- HasSideJob: {model.feature_importances_[0]:.3f}
- MonthlyFamilyIncome: {model.feature_importances_[1]:.3f}

Final Prediction Distribution:
- Predicted Enrolled: {y_pred_full.sum()} students
- Predicted Dropped: {len(y_pred_full) - y_pred_full.sum()} students
"""
        # Display results in the auxiliary panel
        if str(aux_panel) not in left_vpaned.panes():
            left_vpaned.add(aux_panel)
        aux_var.set(True)

        # Clear previous contents in aux_panel (keep header)
        for widget in aux_panel.winfo_children():
            if widget != aux_header:
                widget.destroy()

        # Add text widget to aux_panel
        text_frame = tk.Frame(aux_panel, bg="#1E1E2F")
        text_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=(0,10))

        results_text = tk.Text(text_frame, wrap=tk.WORD, bg="#282A36", fg="#F8F8F2", font=("Arial", 10))
        results_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        scrollbar = ttk.Scrollbar(text_frame, orient=tk.VERTICAL, command=results_text.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        results_text.configure(yscrollcommand=scrollbar.set)

        results_text.insert(tk.END, result_msg)
        results_text.config(state=tk.DISABLED)

        # Show confusion matrix with better visualization
        cm = confusion_matrix(y, y_pred_full)
        fig, ax = plt.subplots(figsize=(7, 5))

        # Create the heatmap
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=['Predicted\nDropped', 'Predicted\nEnrolled'],
                yticklabels=['Actual\nDropped', 'Actual\nEnrolled'],
                cbar_kws={'label': 'Number of Students'},
                annot_kws={'size': 12, 'weight': 'bold'})

        # Customize the plot
        ax.set_xlabel('\nPredicted Status', fontsize=12, fontweight='bold')
        ax.set_ylabel('Actual Status\n', fontsize=12, fontweight='bold')
        ax.set_title('Confusion Matrix - Student Enrollment Prediction\n', fontsize=14, fontweight='bold')

        # Add a grid to separate the quadrants
        for i in range(3):
            ax.axhline(i, color='white', linewidth=2)
            ax.axvline(i, color='white', linewidth=2)

        # Add interpretation text
        interpretation = (
            "Interpretation:\n"
            "• Top-Left: Actually Dropped, Correctly Predicted ✓\n"
            "• Top-Right: Actually Dropped, Wrongly Predicted as Enrolled ✗\n"
            "• Bottom-Left: Actually Enrolled, Wrongly Predicted as Dropped ✗\n"
            "• Bottom-Right: Actually Enrolled, Correctly Predicted ✓"
        )

        ax.text(2.5, 0.5, interpretation, transform=ax.transAxes, fontsize=10,
                verticalalignment='center', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))

        plt.tight_layout()

        for widget in visualization_frame.winfo_children():
            widget.destroy()
        canvas = FigureCanvasTkAgg(fig, master=visualization_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill='both', expand=True)  # Fixed: removed extra parenthesis

        # Update predictions panel
        try:
            results_df = X.copy()
            results_df['Actual_Status'] = y.map({0: 'Dropped', 1: 'Enrolled'})
            results_df['Predicted_Status'] = y_pred_full
            results_df['Predicted_Status'] = results_df['Predicted_Status'].map({0: 'Dropped', 1: 'Enrolled'})
            results_df = results_df.reset_index()

            if str(predict_panel) not in right_vpaned.panes():
                right_vpaned.add(predict_panel)
            pred_var.set(True)

            cols = ['index', 'HasSideJob', 'MonthlyFamilyIncome', 'Actual_Status', 'Predicted_Status']
            predict_tree.config(columns=cols)
            for c in cols:
                predict_tree.heading(c, text=c)
                predict_tree.column(c, width=100, anchor='center')
            
            for item in predict_tree.get_children():
                predict_tree.delete(item)
            
            for i, row in results_df.iterrows():
                vals = [i, row['HasSideJob'], f"{row['MonthlyFamilyIncome']:.3f}", 
                       row['Actual_Status'], row['Predicted_Status']]
                predict_tree.insert('', 'end', values=vals)
                
        except Exception as e:
            print(f"Error updating predictions panel: {e}")

    except Exception as e:
        messagebox.showerror("Prediction Error", f"An error occurred: {e}")

def sort_data():
    try:
        column = column_var.get()
        if column not in df.columns:
            messagebox.showerror("Error", f"Column '{column}' not found!")
            return

        df_sorted = df.sort_values(by=column)
        # Update the Treeview with sorted data
        for row in data_tree.get_children():
            data_tree.delete(row)
        for row in df_sorted.itertuples(index=False):
            data_tree.insert("", "end", values=row)
        messagebox.showinfo("Success", f"Data sorted by {column}")
    except Exception as e:
        messagebox.showerror("Error", f"An error occurred while sorting: {e}")


def search_data():
    query = search_var.get().lower()  # Convert to lowercase for case-insensitive comparison
    for item in data_tree.get_children():
        data_tree.delete(item)
    for item in df.index:
        # Check if any value in the row contains the query string (partial match)
        if any(query in str(value).lower() for value in df.loc[item].values):
            data_tree.insert('', 'end', values=list(df.loc[item]))
    if data_tree.get_children():
        messagebox.showinfo("Search Results", f"Found {len(data_tree.get_children())} matching records")
    else:
        messagebox.showerror("Error", "No results found")


# Function to clean null data
def clean_null_data():
    try:
        global df
        df = df.dropna()  # Drop rows with null values
        messagebox.showinfo("Success", "Null data cleaned successfully!")
    except Exception as e:
        messagebox.showerror("Error", f"An error occurred while cleaning data: {e}")

# Function for aggregation
def show_aggregation_options():
    if not df.empty:
        options = ["SUM", "AVERAGE", "COUNT", "MIN", "MAX"]
        selected_option = simpledialog.askstring("Aggregation", f"Choose an option: {', '.join(options)}")
        if selected_option and selected_option.upper() in options:
            result = None
            if selected_option.upper() == "SUM":
                result = df.sum(numeric_only=True)
            elif selected_option.upper() == "AVERAGE":
                result = df.mean(numeric_only=True)
            elif selected_option.upper() == "COUNT":
                result = df.count()
            elif selected_option.upper() == "MIN":
                result = df.min(numeric_only=True)
            elif selected_option.upper() == "MAX":
                result = df.max(numeric_only=True)
            messagebox.showinfo("Aggregation Result", result.to_string())
        else:
            messagebox.showerror("Invalid Option", "Please select a valid aggregation option.")
    return

def show_descriptive_statistics():
    if not df.empty:
        result = df.describe(include="all").T  # Transpose for better readability

        # Create a new Toplevel window for displaying the table
        stats_window = tk.Toplevel()
        stats_window.title("Descriptive Statistics")
        stats_window.geometry("800x400")

            # Make the window transparent
        stats_window.attributes('-alpha', 0.9)  # Set transparency level (0.0 to 1.0)

        # Create a Treeview widget
        tree = ttk.Treeview(stats_window)
        tree.pack(fill=tk.BOTH, expand=True)

        # Set up columns
        tree["columns"] = result.columns.tolist()
        tree["show"] = "headings"  # Hide the default first column

        # Configure column headers
        for col in result.columns:
            tree.heading(col, text=col)
            tree.column(col, anchor="center", width=100)

        # Insert rows of the DataFrame
        for index, row in result.iterrows():
            tree.insert("", "end", values=[index] + row.tolist())

        # Add a scrollbar for better navigation
        scrollbar = ttk.Scrollbar(stats_window, orient=tk.VERTICAL, command=tree.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        tree.configure(yscrollcommand=scrollbar.set)

        # Add a Done button
        done_button = tk.Button(stats_window, text="Done", command=stats_window.destroy, bg="#81c784",
                                font=("Arial", 12))
        done_button.pack(pady=10)

    else:
        messagebox.showerror("Error", "No data available for statistics.")

def show_data_window(data, title):
    window = tk.Toplevel()
    window.title(title)
    window.geometry("800x400")

    # Get screen dimensions
    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()

    # Calculate the position to center the terms window
    window_width = 800  # Desired width of the terms window
    window_height = 400  # Desired height of the terms window
    center_x = int(screen_width / 2 - window_width / 2)
    center_y = int(screen_height / 2 - window_height / 2)

    # Apply the geometry to the terms window
    window.geometry(f"{window_width}x{window_height}+{center_x}+{center_y}")
    window.attributes('-alpha', 0.9)  # Set transparency level (0.0 to 1.0)
    
    tree = ttk.Treeview(window)
    tree.pack(fill=tk.BOTH, expand=True)

    tree["columns"] = data.columns.tolist()
    tree["show"] = "headings"

    for col in data.columns:
        tree.heading(col, text=col)
        tree.column(col, anchor="center", width=100)

    for row in data.itertuples(index=False):
        tree.insert("", "end", values=row)

    scrollbar = ttk.Scrollbar(window, orient=tk.VERTICAL, command=tree.yview)
    scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
    tree.configure(yscrollcommand=scrollbar.set)

    # Create a button to reset the filtered data
    reset_button = tk.Button(window, text="Reset", command=lambda: reset_data(window), bg="#FF6F61", font=("Arial", 12))
    reset_button.pack(pady=10)

    done_button = tk.Button(window, text="Done", command=window.destroy, bg="#81c784", font=("Arial", 12))
    done_button.pack(pady=10)

def reset_data(window):
    window.destroy()  # Close the data window

def show_top_values():
    if not df.empty:
        column = column_var.get()  # Get the selected column
        if column not in df.columns:
            messagebox.showerror("Error", "Please select a valid column.")
            return

        # Prompt user for the number of top rows
        try:
            n = int(simpledialog.askstring("Top Values", "Enter the number of top values to display:"))
            top_values = df.nlargest(n, columns=column)  # Get top N rows

            # Display in a new window
            show_data_window(top_values, f"Top {n} Values in {column}")
        except ValueError:
            messagebox.showerror("Error", "Please enter a valid number.")
    else:
        messagebox.showerror("Error", "No data available.")

def show_least_values():
    if not df.empty:
        column = column_var.get()  # Get the selected column
        if column not in df.columns:
            messagebox.showerror("Error", "Please select a valid column.")
            return

        # Prompt user for the number of least rows
        try:
            n = int(simpledialog.askstring("Least Values", "Enter the number of least values to display:"))
            least_values = df.nsmallest(n, columns=column)  # Get least N rows

            # Display in a new window
            show_data_window(least_values, f"Least {n} Values in {column}")
        except ValueError:
            messagebox.showerror("Error", "Please enter a valid number.")
    else:
        messagebox.showerror("Error", "No data available.")

def predictions():
    if not df.empty:
        messagebox.showinfo("Info", "Prediction feature coming soon!")
    else:
        messagebox.showerror("Error", "No data available for predictions.")

def filter_data():
    if not df.empty:
        try:
            # Create a custom top-level window for filtering
            filter_window = tk.Toplevel()
            filter_window.title("Filter Data")
            filter_window.geometry("270x120")
            filter_window.resizable(False, False)

            # Get screen dimensions
            screen_width = root.winfo_screenwidth()
            screen_height = root.winfo_screenheight()

            # Calculate the position to center the terms window
            window_width = 270  # Desired width of the terms window
            window_height = 120  # Desired height of the terms window
            center_x = int(screen_width / 2 - window_width / 2)
            center_y = int(screen_height / 2 - window_height / 2)

            # Apply the geometry to the terms window
            filter_window.geometry(f"{window_width}x{window_height}+{center_x}+{center_y}")
            filter_window.attributes('-alpha', 0.8)  # Set transparency level (0.0 to 1.0)

            # Function to handle filtering
            def apply_filter():
                column = column_entry.get()
                if column is None or column.strip() == "":
                    messagebox.showerror("Error", "No column inputted!")
                    return
                
                # Check for case-insensitive column name matching
                column_lower = column.lower()
                matched_columns = [col for col in df.columns if col.lower() == column_lower]
                if not matched_columns:
                    messagebox.showerror("Error", f"Column '{column}' not found!")
                    return
                
                column = matched_columns[0]  # Use the matched column

                value = value_entry.get()
                if value is None or value.strip() == "":
                    messagebox.showinfo("Info", "No value entered for filtering.")
                    return
                
                value_lower = value.lower()  # Convert the input value to lowercase

                # Check if the value exists in the specified column
                if not df[column].astype(str).str.contains(value_lower).any():
                    messagebox.showerror("Error", f"Value '{value}' not found in column '{column}'")
                    return
                
                # Filter the DataFrame based on the specified column and value
                df_filtered = df[df[column].astype(str).str.contains(value_lower)]
                show_data_window(df_filtered, f"Filtered Data: {column} contains '{value}'")
                filter_window.destroy()  # Close the filter window after filtering

            # Function to reset the input fields
            def back_fields():
                filter_window.destroy()

            # Create input fields for column and value
            tk.Label(filter_window, text="Enter column to filter:").grid(row=0, column=0, pady=5)
            column_entry = tk.Entry(filter_window)
            column_entry.grid(row=0, column=1, pady=5)

            tk.Label(filter_window, text="Enter value to filter:").grid(row=1, column=0, pady=5)
            value_entry = tk.Entry(filter_window)
            value_entry.grid(row=1, column=1, pady=5)

            # Create a button to apply the filter
            apply_button = tk.Button(filter_window, text="Apply Filter", command=apply_filter)
            apply_button.grid(row=2, column=0, pady=5)

            # Create a button to reset the input fields
            reset_button = tk.Button(filter_window, text="Back", command=back_fields)
            reset_button.grid(row=2, column=1, pady=5)

            # Run the filter window
            filter_window.transient(root)  # Make the filter window transient to the main window
            filter_window.grab_set()  # Make the filter window modal
            filter_window.focus_set()  # Focus on the filter window

        except Exception as e:
            messagebox.showerror("Invalid data", str(e))

# Dropdown Button Style Update
def configure_dropdown(menu):
    menu.configure(background="#6272A4", foreground="#F8F8F2", font=("Arial", 12), activebackground="#44475A")


def recommend_faculty():
    """Ask user for a subject and show top recommended faculty members."""
    try:
        if 'df' not in globals() or df is None or df.empty:
            messagebox.showerror("Error", "No data available.")
            return

        if ner_module is None:
            messagebox.showerror("spaCy Missing", "spaCy or the NER helper module is not available. Install spaCy and the model (see requirements).")
            return

        subject = simpledialog.askstring("Recommend Faculty", "Enter subject/topic to find best faculty for:")
        if not subject:
            return

        results = ner_module.rank_faculty(df, subject, top_n=10)

        # Safe UI updates
        def safe_ui_updates():
            try:
                # Ensure recommendations panel is visible
                if str(recommend_panel) not in right_vpaned.panes():
                    right_vpaned.add(recommend_panel)
                rec_var.set(True)

                # Clear existing entries safely
                for item in recommend_tree.get_children():
                    recommend_tree.delete(item)

                # Store results globally so selection handler can access them
                global recommend_results
                recommend_results = results  # Store the entire results list

                # Insert new items safely
                for r in recommend_results:
                    name = r.get("name", "")
                    score = f"{r.get('score', 0):.2f}"
                    recommend_tree.insert("", "end", values=(name, score))

                # Update relationship label with top score summary safely
                if relationship_label.winfo_exists():
                    if recommend_results:
                        top = recommend_results[0]
                        relationship_label.config(text=f"Top match: {top.get('name')} ({top.get('score'):.2f})")
                    else:
                        relationship_label.config(text="No recommendations found.")

            except Exception as ui_error:
                print(f"UI update error in recommend_faculty: {ui_error}")
                # Fallback: just show a simple message
                try:
                    if recommend_results:
                        messagebox.showinfo("Recommendations", f"Found {len(recommend_results)} recommendations!")
                    else:
                        messagebox.showinfo("Recommendations", "No recommendations found.")
                except:
                    pass

        # Schedule UI updates to run in the main thread
        root.after(0, safe_ui_updates)

    except Exception as e:
        # Handle the error in the main thread
        def show_error():
            messagebox.showerror("Error", f"An error occurred: {e}")
        root.after(0, show_error)

# Initialize GUI
# Note: theme persistence and sash animations are currently disabled.

# Use ttkbootstrap Window when available for nicer widgets
if tb:
    root = tb.Window(themename='darkly')
else:
    root = tk.Tk()
root.title("Data Analysis Dashboard")
root.geometry("1920x1080")
root.state("zoomed")
root.configure(bg="#1E1E2F")  # Dark background for a modern aesthetic

# Paned Window
paned_window = tk.PanedWindow(root, orient=tk.HORIZONTAL, sashwidth=5, bg="#1E1E2F")
paned_window.pack(fill=tk.BOTH, expand=True)

# Left Frame (Data Table & Controls)
left_frame = tk.Frame(paned_window, bg="#282A36")  
paned_window.add(left_frame)
paned_window.paneconfigure(left_frame, minsize=400)

# Right Frame (Visualization)
right_frame = tk.Frame(paned_window, bg="#282A36")
paned_window.add(right_frame)
paned_window.paneconfigure(right_frame, minsize=600)

# Treeview (Data Table) with both scrollbars inside a frame
columns = ["A", "B", "C", "D"]
data_tree_frame = tk.Frame(left_frame, bg="#282A36")
data_tree_frame.pack(fill=tk.BOTH, expand=True)

data_tree = ttk.Treeview(data_tree_frame, columns=columns, show="headings", style="Custom.Treeview")
vsb = ttk.Scrollbar(data_tree_frame, orient=tk.VERTICAL, command=data_tree.yview)
hsb = ttk.Scrollbar(data_tree_frame, orient=tk.HORIZONTAL, command=data_tree.xview)
data_tree.configure(yscrollcommand=vsb.set, xscrollcommand=hsb.set)
data_tree.grid(row=0, column=0, sticky="nsew")
vsb.grid(row=0, column=1, sticky="ns")
hsb.grid(row=1, column=0, sticky="ew")
data_tree_frame.grid_rowconfigure(0, weight=1)
data_tree_frame.grid_columnconfigure(0, weight=1)

# Search Feature
search_frame = tk.Frame(left_frame, bg="#282A36")
search_frame.pack(fill=tk.X)
search_label = tk.Label(search_frame, text="Search", bg="#44475A", fg="#F8F8F2", font=("Arial", 12))
search_label.pack(side=tk.LEFT, padx=5, pady=5)
search_var = tk.StringVar()
search_entry = tk.Entry(search_frame, textvariable=search_var, font=("Arial", 12), bg="#44475A", fg="#F8F8F2", insertbackground="#F8F8F2")
search_entry.pack(side=tk.LEFT, padx=5, pady=5, fill=tk.X, expand=True)
search_button = tk.Button(search_frame, text="Search", command=lambda: print("Search"), bg="#6272A4", fg="#F8F8F2", font=("Arial", 12))
search_button.pack(side=tk.RIGHT, padx=5, pady=5)


# Controls container (collapsible categories)
data_controls_frame = tk.Frame(left_frame, bg="#282A36")
data_controls_frame.pack(fill=tk.BOTH, side=tk.TOP, padx=6, pady=6)

# Small left-panel collapse/restore toggle
left_restore_btn = None
def toggle_left_panel():
    global left_restore_btn
    try:
        if str(left_frame) in paned_window.panes():
            # hide left frame
            paned_window.forget(left_frame)
            # place a small restore button at the left edge
            left_restore_btn = tk.Button(root, text="▶", bg="#6272A4", fg="#F8F8F2", command=toggle_left_panel)
            left_restore_btn.place(x=0, y=120, width=28, height=60)
        else:
            # restore left frame
            if left_restore_btn:
                try:
                    left_restore_btn.destroy()
                except Exception:
                    pass
                left_restore_btn = None
            # add left_frame back to paned_window (before right_frame if possible)
            try:
                paned_window.add(left_frame)
            except Exception:
                pass
    except Exception:
        pass

# Left sidebar (VS Code-style) and category panels
left_sidebar = tk.Frame(left_frame, width=56, bg="#23232B")
left_sidebar.pack(side=tk.LEFT, fill=tk.Y)

# Small helper to create narrow sidebar buttons
def _sb_button(text, command):
    btn = tk.Button(left_sidebar, text=text, width=6, bg="#2E2F3A", fg="#F8F8F2", relief=tk.FLAT, command=command)
    btn.pack(pady=6)
    return btn

# Controls area to the right of the sidebar
controls_pane = tk.Frame(left_frame, bg="#282A36")
controls_pane.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

# Recreate the data tree inside the controls pane so it sits to the right of the sidebar
try:
    data_tree_frame.destroy()
except Exception:
    pass
data_tree_frame = tk.Frame(controls_pane, bg="#282A36")
data_tree_frame.pack(fill=tk.BOTH, expand=True)

data_tree = ttk.Treeview(data_tree_frame, columns=columns, show="headings", style="Custom.Treeview")
vsb = ttk.Scrollbar(data_tree_frame, orient=tk.VERTICAL, command=data_tree.yview)
hsb = ttk.Scrollbar(data_tree_frame, orient=tk.HORIZONTAL, command=data_tree.xview)
data_tree.configure(yscrollcommand=vsb.set, xscrollcommand=hsb.set)
data_tree.grid(row=0, column=0, sticky="nsew")
vsb.grid(row=0, column=1, sticky="ns")
hsb.grid(row=1, column=0, sticky="ew")
data_tree_frame.grid_rowconfigure(0, weight=1)
data_tree_frame.grid_columnconfigure(0, weight=1)

# Recreate the search frame inside controls_pane as well
try:
    search_frame.destroy()
except Exception:
    pass
search_frame = tk.Frame(controls_pane, bg="#282A36")
search_frame.pack(fill=tk.X)
search_label = tk.Label(search_frame, text="Search", bg="#44475A", fg="#F8F8F2", font=("Arial", 12))
search_label.pack(side=tk.LEFT, padx=5, pady=5)
search_var = tk.StringVar()
search_entry = tk.Entry(search_frame, textvariable=search_var, font=("Arial", 12), bg="#44475A", fg="#F8F8F2", insertbackground="#F8F8F2")
search_entry.pack(side=tk.LEFT, padx=5, pady=5, fill=tk.X, expand=True)
search_button = tk.Button(search_frame, text="Search", command=lambda: print("Search"), bg="#6272A4", fg="#F8F8F2", font=("Arial", 12))
search_button.pack(side=tk.RIGHT, padx=5, pady=5)

# Category frames (only one visible at a time)
file_frame = tk.Frame(controls_pane, bg="#282A36")
analysis_frame = tk.Frame(controls_pane, bg="#282A36")
dataops_frame = tk.Frame(controls_pane, bg="#282A36")
values_frame = tk.Frame(controls_pane, bg="#282A36")

category_frames = {
    'Files': file_frame,
    'Analysis': analysis_frame,
    'Data': dataops_frame,
    'Quick': values_frame,
}

def show_category(name):
    # hide all
    for f in category_frames.values():
        f.pack_forget()
    # show selected
    category_frames[name].pack(fill=tk.BOTH, expand=True)

# Sidebar buttons
_sb_button('Files', lambda: show_category('Files'))
_sb_button('Analysis', lambda: show_category('Analysis'))
_sb_button('Data', lambda: show_category('Data'))
_sb_button('Quick', lambda: show_category('Quick'))

# Button style
button_style = {"bg": "#6272A4", "fg": "#F8F8F2", "font": ("Arial", 12)}

# Populate Files category
upload_button = tk.Button(file_frame, text="Upload CSV", command=upload_file, **button_style)
upload_button.pack(fill=tk.X, pady=4, padx=6)
filter_button = tk.Button(file_frame, text="Filter Data", command=filter_data, **button_style)
filter_button.pack(fill=tk.X, pady=4, padx=6)

# Populate Analysis category
analyze_button = tk.Button(analysis_frame, text="Analyze Relationship", command=analyze_relationship, state=tk.DISABLED, **button_style)
analyze_button.pack(fill=tk.X, pady=4, padx=6)
visualize_button = tk.Button(analysis_frame, text="Visualize Data", command=visualize_data, state=tk.DISABLED, **button_style)
visualize_button.pack(fill=tk.X, pady=4, padx=6)
recommend_button = tk.Button(analysis_frame, text="Recommend", command=recommend_faculty, state=tk.DISABLED, **button_style)
recommend_button.pack(fill=tk.X, pady=4, padx=6)

# Populate Data category
agg_button = tk.Button(dataops_frame, text="Aggregations", command=show_aggregation_options, **button_style)
agg_button.pack(fill=tk.X, pady=4, padx=6)
stats_button = tk.Button(dataops_frame, text="Descriptive Statistics", command=show_descriptive_statistics, **button_style)
stats_button.pack(fill=tk.X, pady=4, padx=6)
clean_button = tk.Button(dataops_frame, text="Clean Null Data", command=clean_null_data, **button_style)
clean_button.pack(fill=tk.X, pady=4, padx=6)
save_button = tk.Button(dataops_frame, text="Save Visualization", command=save_visualization, state=tk.DISABLED, **button_style)
save_button.pack(fill=tk.X, pady=4, padx=6)
predictions_button = tk.Button(dataops_frame, text="Predictions", command=Predictions, **button_style)
predictions_button.pack(fill=tk.X, pady=4, padx=6)
cluster_students_button = tk.Button(dataops_frame, text="Cluster Students", command=cluster_students, **button_style)
cluster_students_button.pack(fill=tk.X, pady=4, padx=6)
cluster_faculty_button = tk.Button(dataops_frame, text="Cluster Faculty", command=cluster_faculty, **button_style)
cluster_faculty_button.pack(fill=tk.X, pady=4, padx=6)


# Populate Quick category
top_button = tk.Button(values_frame, text="Top Values", command=show_top_values, **button_style)
top_button.pack(fill=tk.X, pady=4, padx=6)
least_button = tk.Button(values_frame, text="Least Values", command=show_least_values, **button_style)
least_button.pack(fill=tk.X, pady=4, padx=6)
sort_label = tk.Label(values_frame, text="Sort By", bg="#282A36", fg="#F8F8F2", font=("Arial", 12))
sort_label.pack(anchor='w', pady=(6,0), padx=6)
sort_column_var = tk.StringVar()


sort_menu = ttk.OptionMenu(values_frame, sort_column_var, "")
sort_menu.pack(fill=tk.X, pady=4, padx=6)

# Show a default category
show_category('Files')

# Column Selection & Graph Options moved into Analysis section (use pack to avoid mixing geometry managers)
column_label = tk.Label(analysis_frame, text="Select Column for Graph", bg="#282A36", fg="#F8F8F2", font=("Arial", 12))
column_label.pack(anchor='w', padx=6, pady=(8,0))
column_var = tk.StringVar()
column_menu = ttk.OptionMenu(analysis_frame, column_var, "A")
column_menu.pack(fill='x', padx=6, pady=4)

column2_label = tk.Label(analysis_frame, text="Select Second Column", bg="#282A36", fg="#F8F8F2", font=("Arial", 12))
column2_label.pack(anchor='w', padx=6, pady=(4,0))
column2_var = tk.StringVar()
column2_menu = ttk.OptionMenu(analysis_frame, column2_var, "B")
column2_menu.pack(fill='x', padx=6, pady=4)

# Graph Type Selection
graph_type_label = tk.Label(analysis_frame, text="Graph Type", bg="#282A36", fg="#F8F8F2", font=("Arial", 12))
graph_type_label.pack(anchor='w', padx=6, pady=(6,0))
graph_type_var = tk.StringVar(value="Histogram")
graph_type_menu = ttk.OptionMenu(analysis_frame, graph_type_var, "Histogram", "Histogram", "Bar", "Scatter", "Line")
graph_type_menu.pack(fill='x', padx=6, pady=4)

# Color Picker
color_label = tk.Label(analysis_frame, text="Graph Color", bg="#282A36", fg="#F8F8F2", font=("Arial", 12))
color_label.pack(anchor='w', padx=6, pady=(6,0))
color_var = tk.StringVar(value="blue")
color_menu = ttk.OptionMenu(analysis_frame, color_var, "blue", "blue", "red", "green", "purple", "orange")
color_menu.pack(fill='x', padx=6, pady=4)

# Visibility toggles for View menu (default to visible)
vis_var = tk.BooleanVar(value=True)
aux_var = tk.BooleanVar(value=True)
rec_var = tk.BooleanVar(value=True)
det_var = tk.BooleanVar(value=True)
pred_var = tk.BooleanVar(value=True)

# Theme definitions (light and dark)
THEMES = {
    'dark': {
        'bg': '#1E1E2F',
        'panel': '#282A36',
        'accent': '#6272A4',
        'muted': '#44475A',
        'fg': '#F8F8F2',
        'button_bg': '#6272A4',
        'button_hover': '#4ea36e',
        'tree_bg': '#282A36'
    },
    'light': {
        'bg': '#F5F7FA',
        'panel': '#FFFFFF',
        'accent': '#3A7BD5',
        'muted': '#E6EEF8',
        'fg': '#0F172A',
        'button_bg': '#3A7BD5',
        'button_hover': '#2f66b0',
        'tree_bg': '#FFFFFF'
    }
}

current_theme = 'dark'

def hex_to_rgb(h):
    h = h.lstrip('#')
    return tuple(int(h[i:i+2], 16) for i in (0, 2, 4))

def rgb_to_hex(rgb):
    return '#{:02x}{:02x}{:02x}'.format(*rgb)

def interpolate_color(a, b, t):
    ar, ag, ab = hex_to_rgb(a)
    br, bgc, bb = hex_to_rgb(b)
    return rgb_to_hex((int(ar + (br - ar) * t), int(ag + (bgc - ag) * t), int(ab + (bb - ab) * t)))

def animate_theme_transition(old_bg, new_bg, steps=8, delay=30):
    # Subtle background color animation for root + main panels
    for i in range(steps):
        t = (i + 1) / steps
        color = interpolate_color(old_bg, new_bg, t)
        root.after(int(i * delay), lambda c=color: root.configure(bg=c))
        root.after(int(i * delay), lambda c=color: visualization_frame.configure(bg=c))

def apply_theme(name):
    global current_theme
    if name not in THEMES:
        return
    old = THEMES.get(current_theme, THEMES['dark'])['bg']
    theme = THEMES[name]
    current_theme = name
    # animate bg change
    animate_theme_transition(old, theme['bg'])

    # Apply colors
    root.configure(bg=theme['bg'])
    left_sidebar.configure(bg=theme['panel'])
    controls_pane.configure(bg=theme['panel'])
    data_tree_frame.configure(bg=theme['panel'])
    visualization_frame.configure(bg=theme['bg'])
    right_frame.configure(bg=theme['panel'])

    # update style for Treeview
    try:
        style.configure("Custom.Treeview", background=theme['tree_bg'], fieldbackground=theme['panel'], foreground=theme['fg'])
    except Exception:
        pass

    # update buttons if they exist
    try:
        btn_bg = theme['button_bg']
        for btn in buttons:
            btn.config(bg=btn_bg, fg=theme['fg'])
    except Exception:
        pass

    # update detail text colors
    try:
        detail_text.config(bg=theme['bg'], fg=theme['fg'])
    except Exception:
        pass
    # If ttkbootstrap is available, try to set a matching theme
    try:
        if tb:
            tb_map = {'dark': 'darkly', 'light': 'flatly'}
            tb_theme = tb_map.get(name, 'darkly')
            try:
                tb.Style().theme_use(tb_theme)
            except Exception:
                try:
                    root.style.theme_use(tb_theme)
                except Exception:
                    pass
    except Exception:
        pass


# Replace the grid with nested PanedWindows so panels are resizable with sashes
right_hpaned = tk.PanedWindow(right_frame, orient=tk.HORIZONTAL, sashwidth=6)
right_hpaned.pack(fill=tk.BOTH, expand=True)

# Left and right vertical paned windows inside the horizontal paned
left_vpaned = tk.PanedWindow(right_hpaned, orient=tk.VERTICAL, sashwidth=6)
right_vpaned = tk.PanedWindow(right_hpaned, orient=tk.VERTICAL, sashwidth=6)
right_hpaned.add(left_vpaned)
right_hpaned.add(right_vpaned)

# Top-left: main visualization area
visualization_frame = tk.Frame(left_vpaned, bg="#1E1E2F", bd=0)
left_vpaned.add(visualization_frame)

# header with close button for visualization
viz_header = tk.Frame(visualization_frame, bg="#1E1E2F")
viz_header.pack(fill=tk.X)
viz_label = tk.Label(viz_header, text="Visualization", bg="#1E1E2F", fg="#F8F8F2", font=("Arial", 12))
viz_label.pack(side=tk.LEFT, padx=6, pady=4)
def _close_visualization():
    try:
        left_vpaned.forget(visualization_frame)
        vis_var.set(False)
    except Exception:
        pass
viz_close = tk.Button(viz_header, text="X", bg="#FF6F61", fg="#F8F8F2", width=3, command=_close_visualization)
viz_close.pack(side=tk.RIGHT, padx=6, pady=4)

# Bottom-left: auxiliary visualization / details
aux_panel = tk.Frame(left_vpaned, bg="#1E1E2F")
left_vpaned.add(aux_panel)
aux_header = tk.Frame(aux_panel, bg="#1E1E2F")
aux_header.pack(fill=tk.X)
aux_label = tk.Label(aux_header, text="Auxiliary", bg="#1E1E2F", fg="#F8F8F2", font=("Arial", 12))
aux_label.pack(side=tk.LEFT, padx=6, pady=4)
def _close_aux():
    try:
        left_vpaned.forget(aux_panel)
        aux_var.set(False)
    except Exception:
        pass
aux_close = tk.Button(aux_header, text="X", bg="#FF6F61", fg="#F8F8F2", width=3, command=_close_aux)
aux_close.pack(side=tk.RIGHT, padx=6, pady=4)

# Top-right: recommendations list
recommend_panel = tk.Frame(right_vpaned, bg="#282A36")
right_vpaned.add(recommend_panel)
recommend_panel.grid_rowconfigure(1, weight=1)
recommend_header = tk.Frame(recommend_panel, bg="#282A36")
recommend_header.grid(row=0, column=0, sticky="nsew", padx=8, pady=8)
recommend_title = tk.Label(recommend_header, text="Recommendations", bg="#282A36", fg="#F8F8F2", font=("Arial", 14))
recommend_title.pack(side=tk.LEFT)
def _close_recommend():
    try:
        right_vpaned.forget(recommend_panel)
        rec_var.set(False)
    except Exception:
        pass
recommend_close = tk.Button(recommend_header, text="X", bg="#FF6F61", fg="#F8F8F2", width=3, command=_close_recommend)
recommend_close.pack(side=tk.RIGHT)

# Treeview to show recommended faculty (Name + Score)
recommend_tree = ttk.Treeview(recommend_panel, columns=("Name", "Score"), show="headings")
recommend_tree.heading("Name", text="Name")
recommend_tree.heading("Score", text="Score")
recommend_tree.column("Name", anchor="w")
recommend_tree.column("Score", anchor="center", width=80)
recommend_tree.grid(row=1, column=0, sticky="nsew", padx=8, pady=(0,8))
recommend_scroll = ttk.Scrollbar(recommend_panel, orient=tk.VERTICAL, command=recommend_tree.yview)
recommend_scroll.grid(row=1, column=1, sticky="ns", pady=(0,8))
recommend_tree.configure(yscrollcommand=recommend_scroll.set)

# When a recommendation is selected, show the full bio/text in the detail panel
def on_recommend_select(event):
    try:
        sel = recommend_tree.selection()
        if not sel:
            return
        # Use index to map to last results list (preserves ordering)
        idx = recommend_tree.index(sel[0])
        if 'recommend_results' not in globals() or recommend_results is None:
            return
        if idx < 0 or idx >= len(recommend_results):
            return
        r = recommend_results[idx]

        # Construct detailed information
        details = f"Faculty ID: {r.get('name', '')}\n"
        details += f"Age: {r.get('age', '')}\n"
        details += f"Gender: {r.get('gender', '')}\n"
        details += f"Years of Experience: {r.get('years_experience', '')}\n"
        details += f"Field of Expertise: {r.get('field_of_expertise', '')}\n"
        details += f"Relevance Score: {r.get('score', 0):.2f}"

        # Display in detail_text
        detail_text.config(state=tk.NORMAL)
        detail_text.delete('1.0', tk.END)
        detail_text.insert(tk.END, details)
        detail_text.config(state=tk.DISABLED)
        detail_text.see('1.0')
    except Exception:
        pass

recommend_tree.bind('<<TreeviewSelect>>', on_recommend_select)

# Bottom-right: detail view for selected faculty
detail_panel = tk.Frame(right_vpaned, bg="#282A36")
right_vpaned.add(detail_panel)
detail_header = tk.Frame(detail_panel, bg="#282A36")
detail_header.pack(fill=tk.X)
detail_title = tk.Label(detail_header, text="Details", bg="#282A36", fg="#F8F8F2", font=("Arial", 14))
detail_title.pack(side=tk.LEFT, padx=8, pady=8)
def _close_detail():
    try:
        right_vpaned.forget(detail_panel)
        det_var.set(False)
    except Exception:
        pass
detail_close = tk.Button(detail_header, text="X", bg="#FF6F61", fg="#F8F8F2", width=3, command=_close_detail)
detail_close.pack(side=tk.RIGHT, padx=8, pady=8)

# Text widget to show selected faculty bio or details (with its own scrollbar)
detail_text_frame = tk.Frame(detail_panel, bg="#282A36")
detail_text_frame.pack(fill=tk.BOTH, expand=True, padx=8, pady=(0,8))
detail_text = tk.Text(detail_text_frame, wrap=tk.WORD, bg="#1E1E2F", fg="#F8F8F2", bd=0)
detail_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
detail_scroll = ttk.Scrollbar(detail_text_frame, orient=tk.VERTICAL, command=detail_text.yview)
detail_scroll.pack(side=tk.RIGHT, fill=tk.Y)
detail_text.configure(yscrollcommand=detail_scroll.set)

# Right-most: predictions panel (shows recent prediction results)
predict_panel = tk.Frame(right_vpaned, bg="#282A36")
right_vpaned.add(predict_panel)
predict_header = tk.Frame(predict_panel, bg="#282A36")
predict_header.pack(fill=tk.X)
predict_title = tk.Label(predict_header, text="Predictions", bg="#282A36", fg="#F8F8F2", font=("Arial", 14))
predict_title.pack(side=tk.LEFT, padx=8, pady=8)
def _close_predict():
    try:
        right_vpaned.forget(predict_panel)
        pred_var.set(False)
    except Exception:
        pass
predict_close = tk.Button(predict_header, text="X", bg="#FF6F61", fg="#F8F8F2", width=3, command=_close_predict)
predict_close.pack(side=tk.RIGHT, padx=8, pady=8)

# Treeview to show predictions (features + actual + predicted)
predict_tree = ttk.Treeview(predict_panel, columns=("idx", "actual", "predicted"), show="headings")
predict_tree.heading("idx", text="#")
predict_tree.heading("actual", text="Actual")
predict_tree.heading("predicted", text="Predicted")
predict_tree.column("idx", width=40, anchor="center")
predict_tree.column("actual", anchor="w")
predict_tree.column("predicted", anchor="w")
predict_tree.pack(fill=tk.BOTH, expand=True, padx=8, pady=(0,8))
predict_scroll = ttk.Scrollbar(predict_panel, orient=tk.VERTICAL, command=predict_tree.yview)
predict_scroll.pack(side=tk.RIGHT, fill=tk.Y)
predict_tree.configure(yscrollcommand=predict_scroll.set)

# Relationship label moved to top of visualization frame area
relationship_label = tk.Label(visualization_frame, text="Correlation Results will appear here.", bg="#1E1E2F", fg="#F8F8F2", font=("Arial", 14))
relationship_label.pack(anchor="nw", padx=10, pady=6)

# Toggle helper used by View menu to show/hide panes
def toggle_panel(panel, paned, var):
    try:
        if var.get():
            # show panel if not already added
            if str(panel) not in paned.panes():
                paned.add(panel)
        else:
            # hide panel if present
            if str(panel) in paned.panes():
                paned.forget(panel)
    except Exception:
        pass

# Menu for showing/hiding panels
menubar = tk.Menu(root)
view_menu = tk.Menu(menubar, tearoff=0)
view_menu.add_checkbutton(label="Visualization", variable=vis_var, command=lambda: toggle_panel(visualization_frame, left_vpaned, vis_var))
view_menu.add_checkbutton(label="Auxiliary", variable=aux_var, command=lambda: toggle_panel(aux_panel, left_vpaned, aux_var))
view_menu.add_checkbutton(label="Recommendations", variable=rec_var, command=lambda: toggle_panel(recommend_panel, right_vpaned, rec_var))
view_menu.add_checkbutton(label="Details", variable=det_var, command=lambda: toggle_panel(detail_panel, right_vpaned, det_var))
view_menu.add_checkbutton(label="Predictions", variable=pred_var, command=lambda: toggle_panel(globals().get('predict_panel'), right_vpaned, pred_var))
view_menu.add_separator()

def restore_default_view():
    """Re-add all main panels to their paned windows and set view toggles to True."""
    try:
        # Left side
        if str(visualization_frame) not in left_vpaned.panes():
            left_vpaned.add(visualization_frame)
        if str(aux_panel) not in left_vpaned.panes():
            left_vpaned.add(aux_panel)

        # Right side
        if str(recommend_panel) not in right_vpaned.panes():
            right_vpaned.add(recommend_panel)
        if str(detail_panel) not in right_vpaned.panes():
            right_vpaned.add(detail_panel)
        try:
            p = globals().get('predict_panel')
            if p is not None and str(p) not in right_vpaned.panes():
                right_vpaned.add(p)
        except Exception:
            pass

        # Ensure the vars reflect the visible state
        vis_var.set(True)
        aux_var.set(True)
        rec_var.set(True)
        det_var.set(True)
        try:
            pred_var.set(True)
        except Exception:
            pass
    except Exception:
        pass


# Note: sash animation helpers removed per user choice (Option B).

view_menu.add_command(label="Restore Default Layout", command=restore_default_view)
menubar.add_cascade(label="View", menu=view_menu)
# Theme menu
theme_menu = tk.Menu(menubar, tearoff=0)
theme_menu.add_radiobutton(label="Dark", command=lambda: apply_theme('dark'))
theme_menu.add_radiobutton(label="Light", command=lambda: apply_theme('light'))
menubar.add_cascade(label="Theme", menu=theme_menu)
root.config(menu=menubar)

# Treeview Custom Styles
style = ttk.Style()
style.configure("Custom.Treeview", background="#282A36", foreground="#F8F8F2", rowheight=25, fieldbackground="#282A36")
style.map("Custom.Treeview", background=[("selected", "#6272A4")], foreground=[("selected", "#F8F8F2")])

# Apply default theme
try:
    apply_theme(current_theme)
except Exception:
    pass

# Event Binding for Buttons
upload_button.config(command=upload_file)
visualize_button.config(command=visualize_data)
analyze_button.config(command=analyze_relationship)
agg_button.config(command=show_aggregation_options)
stats_button.config(command=show_descriptive_statistics)
clean_button.config(command=clean_null_data)
save_button.config(command=save_visualization)


search_button.config(command=search_data)
top_button.config(command=show_top_values)
least_button.config(command=show_least_values)
filter_button.config(command=filter_data)

# Function to change button color on hover
def on_enter(e):
    try:
        th = THEMES.get(current_theme, THEMES['dark'])
        e.widget.config(bg=th['button_hover'])
    except Exception:
        e.widget.config(bg="Green")

def on_leave(e):
    try:
        th = THEMES.get(current_theme, THEMES['dark'])
        e.widget.config(bg=th['button_bg'])
    except Exception:
        e.widget.config(bg="#6272A4")

# Function to perform sorting when sort_column_var changes
def sort_on_selection(*args):
    column = sort_column_var.get()
    if column == "Don't sort":
        # Reset the Treeview to unsorted original df
        for row in data_tree.get_children():
            data_tree.delete(row)
        for row in df.itertuples(index=False):
            data_tree.insert("", "end", values=row)
        # No message shown for reset
    elif column and column in df.columns:
        try:
            df_sorted = df.sort_values(by=column)
            # Update the Treeview with sorted data
            for row in data_tree.get_children():
                data_tree.delete(row)
            for row in df_sorted.itertuples(index=False):
                data_tree.insert("", "end", values=row)
            messagebox.showinfo("Success", f"Data sorted by {column}")
        except Exception as e:
            messagebox.showerror("Error", f"An error occurred while sorting: {e}")
    else:
        messagebox.showerror("Error", f"Column '{column}' not found!")

# Bind the trace to detect changes of sort_column_var
sort_column_var.trace_add('write', sort_on_selection)

# Add hover functionality to a button
def add_hover_effect(button):
    button.bind("<Enter>", on_enter)
    button.bind("<Leave>", on_leave)

# Apply hover effect to all buttons
buttons = [upload_button, analyze_button, visualize_button, agg_button, stats_button,
           clean_button, save_button, top_button, least_button, search_button, predictions_button, filter_button]

# Add recommend_button to hoverable buttons
buttons.append(recommend_button)
buttons.extend([cluster_students_button, cluster_faculty_button])

for btn in buttons:
    add_hover_effect(btn)

def on_close():
    if messagebox.askyesno("Quit", "Do you really want to close the application?"):
        root.destroy()
        exit()

root.protocol("WM_DELETE_WINDOW", on_close)

# Run the application
root.mainloop()

