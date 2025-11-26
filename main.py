import tkinter as tk
from tkinter import filedialog, ttk, messagebox
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from tkinter import simpledialog


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
            column2_var.set(columns[0])  # Default to the first column
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
        except Exception as e:
            messagebox.showerror("Error", f"An error occurred: {e}")

def visualize_data():
    try:
        column = column_var.get()
        graph_type = graph_type_var.get()
        color = color_var.get()

        # Check if the selected column is valid
        if column not in df.columns:
            messagebox.showerror("Error", f"Column '{column}' not found!")
            return

        # Convert string columns to integers if needed
        if df[column].dtype == 'object':  # If it's a string/categorical column
            df[column] = df[column].astype("category").cat.codes
            messagebox.showinfo("Conversion", f"Column '{column}' converted to numeric codes.")

        # Check if the column is numeric
        if not pd.api.types.is_numeric_dtype(df[column]):
            messagebox.showerror("Error", f"Column '{column}' must be numeric for this graph type.")
            return

        # Create a new figure for visualization
        fig, ax = plt.subplots(figsize=(8, 6))

        # Set figure and axes background color
        fig.patch.set_facecolor("#1E1E2F")  # Matches the app background
        ax.set_facecolor("#1E1E2F")  # Matches the app background

        # Visualization logic based on the selected graph type
        if graph_type == "Histogram":
            sns.histplot(df[column], kde=True, color=color, ax=ax)
            ax.set_title(f"Histogram of {column}")
            ax.set_xlabel(column)

        elif graph_type == "Bar":
            sns.countplot(y=df[column], color=color, ax=ax)
            ax.set_title(f"Bar Graph of {column}")
            ax.set_ylabel(column)

        elif graph_type == "Scatter":
            if len(df.columns) > 1:
                x_column = df.columns[0]

                if not pd.api.types.is_numeric_dtype(df[x_column]):
                    messagebox.showerror("Error", f"Column '{x_column}' must be numeric for scatter plot.")
                    return
                sns.scatterplot(data=df, x=x_column, y=column, color=color, ax=ax)
                ax.set_title(f"Scatter Plot of {column} vs {x_column}")
                ax.set_xlabel(x_column)
                ax.set_ylabel(column)
            else:
                messagebox.showerror("Error", "Scatter plot requires at least two columns.")
                return

        elif graph_type == "Line":
            df[column].plot(kind="line", color=color, ax=ax)
            ax.set_title(f"Line Graph of {column}")
            ax.set_xlabel(column)

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


# Initialize GUI
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

# Treeview (Data Table)
columns = ["A", "B", "C", "D"]
data_tree = ttk.Treeview(left_frame, columns=columns, show="headings", style="Custom.Treeview")
data_tree.pack(fill=tk.BOTH, expand=True)

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

# Data Controls Frame
data_controls_frame = tk.Frame(left_frame, bg="#282A36")
data_controls_frame.pack(fill=tk.X)

# Button Style
button_style = {"bg": "#6272A4", "fg": "#F8F8F2", "font": ("Arial", 12)}

# Buttons for Controls
upload_button = tk.Button(data_controls_frame, text="Upload CSV", command=lambda: print("Upload CSV"), **button_style)
upload_button.grid(row=0, column=0, padx=10, pady=10)

analyze_button = tk.Button(data_controls_frame, text="Analyze Relationship", command=lambda: print("Analyze"), state=tk.DISABLED, **button_style)
analyze_button.grid(row=0, column=1, padx=10, pady=10)

visualize_button = tk.Button(data_controls_frame, text="Visualize Data", command=lambda: print("Visualize"), state=tk.DISABLED, **button_style)
visualize_button.grid(row=0, column=2, padx=10, pady=10)

sort_label = tk.Label(data_controls_frame, text="Sort By", bg="#282A36", fg="#F8F8F2", font=("Arial", 12))
sort_label.grid(row=1, column=2, padx=10, pady=(10), sticky="w") 
sort_column_var = tk.StringVar()
sort_menu = ttk.OptionMenu(data_controls_frame, sort_column_var, "")
sort_menu.grid(row=1, column=3, padx=10, pady=10)

agg_button = tk.Button(data_controls_frame, text="Aggregations", command=lambda: print("Aggregations"), **button_style)
agg_button.grid(row=0, column=3, padx=10, pady=10)

stats_button = tk.Button(data_controls_frame, text="Descriptive Statistics", command=lambda: print("Descriptive Stats"), **button_style)
stats_button.grid(row=1, column=4, padx=10, pady=10)

clean_button = tk.Button(data_controls_frame, text="Clean Null Data", command=lambda: print("Clean"), **button_style)
clean_button.grid(row=1, column=0, padx=10, pady=10)

save_button = tk.Button(data_controls_frame, text="Save Visualization", command=save_visualization, state=tk.DISABLED, **button_style)
save_button.grid(row=1, column=1, padx=10, pady=10)

predictions_button = tk.Button(data_controls_frame, text="Predictions", command=predictions, **button_style)
predictions_button.grid(row=0, column=4, padx=10, pady=10)

# Add a button for filtering
filter_button = tk.Button(data_controls_frame, text="Filter Data", command=filter_data, **button_style)
filter_button.grid(row=5, column=2, padx=10, pady=10)

top_button = tk.Button(data_controls_frame, text="Top Values", command=lambda: print("Top Values"), **button_style)
top_button.grid(row=5, column=0, padx=10, pady=10)

least_button = tk.Button(data_controls_frame, text="Least Values", command=lambda: print("Least Values"), **button_style)
least_button.grid(row=5, column=1, padx=10, pady=10)

# Column Selection for Analysis
column_label = tk.Label(data_controls_frame, text="Select Column for Graph", bg="#282A36", fg="#F8F8F2", font=("Arial", 12))
column_label.grid(row=2, column=0, padx=10, pady=10)
column_var = tk.StringVar()
column_menu = ttk.OptionMenu(data_controls_frame, column_var, "A")
column_menu.grid(row=2, column=1, padx=10, pady=10)

column2_label = tk.Label(data_controls_frame, text="Select Second Column", bg="#282A36", fg="#F8F8F2", font=("Arial", 12))
column2_label.grid(row=2, column=2, padx=10, pady=10)
column2_var = tk.StringVar()
column2_menu = ttk.OptionMenu(data_controls_frame, column2_var, "B")
column2_menu.grid(row=2, column=3, padx=10, pady=10)

# Graph Type Selection
graph_type_label = tk.Label(data_controls_frame, text="Graph Type", bg="#282A36", fg="#F8F8F2", font=("Arial", 12))
graph_type_label.grid(row=4, column=0, padx=10, pady=10)
graph_type_var = tk.StringVar(value="Histogram")
graph_type_menu = ttk.OptionMenu(data_controls_frame, graph_type_var, "Histogram", "Histogram", "Bar", "Scatter", "Line")
graph_type_menu.grid(row=4, column=1, padx=10, pady=10)

# Color Picker
color_label = tk.Label(data_controls_frame, text="Graph Color", bg="#282A36", fg="#F8F8F2", font=("Arial", 12))
color_label.grid(row=4, column=2, padx=10, pady=10)
color_var = tk.StringVar(value="blue")
color_menu = ttk.OptionMenu(data_controls_frame, color_var, "blue", "blue", "red", "green", "purple", "orange")
color_menu.grid(row=4, column=3, padx=10, pady=10)

# Visualization Frame
visualization_frame = tk.Frame(right_frame, bg="#1E1E2F")
visualization_frame.pack(fill=tk.BOTH, expand=True)

# Relationship Label
relationship_label = tk.Label(right_frame, text="Correlation Results will appear here.", bg="#1E1E2F", fg="#F8F8F2", font=("Arial", 14))
relationship_label.pack(padx=10, pady=10)

# Treeview Custom Styles
style = ttk.Style()
style.configure("Custom.Treeview", background="#282A36", foreground="#F8F8F2", rowheight=25, fieldbackground="#282A36")
style.map("Custom.Treeview", background=[("selected", "#6272A4")], foreground=[("selected", "#F8F8F2")])

# Event Binding for Buttons
upload_button.config(command=upload_file)
visualize_button.config(command=visualize_data)
analyze_button.config(command=analyze_relationship)
agg_button.config(command=show_aggregation_options)
stats_button.config(command=show_descriptive_statistics)
clean_button.config(command=clean_null_data)
save_button.config(command=save_visualization)
# Remove old event binding for sort_button
# sort_button.config(command=sort_data)
search_button.config(command=search_data)
top_button.config(command=show_top_values)
least_button.config(command=show_least_values)
filter_button.config(command=filter_data)

# Function to change button color on hover
def on_enter(e):
    e.widget.config(bg="Green")

def on_leave(e):
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

for btn in buttons:
    add_hover_effect(btn)

def on_close():
    if messagebox.askyesno("Quit", "Do you really want to close the application?"):
        root.destroy()
        exit()

root.protocol("WM_DELETE_WINDOW", on_close)

# Run the application
root.mainloop()

