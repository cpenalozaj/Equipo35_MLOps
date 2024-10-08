import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix


class DataExplorer:

    @staticmethod
    def explore_data(data):
        """
        Prints the first 5 rows, the data description and the data info.

        Parameters:
        data: The dataframe to explore

        Returns:
        None

        """
        print(data.head().T)
        print(data.describe())
        print(data.info())

    @staticmethod
    def plot_histograms(dataframe, column_names):
        """
        Plots histograms for the specified columns in the dataframe.

        Parameters:
        dataframe: The dataframe containing the data to plot
        column_names: A list of column names to plot

        Returns:
        None

        """
        num_columns = len(column_names)
        num_rows = (num_columns + 1) // 2  # Calculate the number of rows needed
        sns.set(style="whitegrid")
        fig, axs = plt.subplots(num_rows, 3, figsize=(12, num_rows * 5))
        axs = axs.flatten()

        for i, column in enumerate(column_names):
            sns.countplot(x=column, data=dataframe, ax=axs[i], hue=column, dodge=False, palette="Set2", legend=False)
            axs[i].set_title(f'Gráfico de barras de recuento para {column}')
            axs[i].set_xlabel(column)
            axs[i].set_ylabel('Frecuencia')
            axs[i].tick_params(axis='x', rotation=45)

        # Remove any unused axes if necessary
        for j in range(i + 1, len(axs)):
            fig.delaxes(axs[j])

        plt.tight_layout()
        plt.show()

    @staticmethod
    def plot_confusion_matrix(y_true, y_pred, labels=('Best', 'Good', 'Pass', 'Vg')):
        """
        This function plots a confusion matrix using Seaborn's heatmap with colors, rectangles, and labels.

        Parameters:
        - y_true: The true labels
        - y_pred: The predicted labels
        - labels: A list of labels for the x and y axes (classes)

        Example usage:
        plot_confusion_matrix(y_test, y_pred, labels=['Best', 'Good', 'Pass', 'Vg'])
        """

        # Generate confusion matrix
        cm = confusion_matrix(y_true, y_pred)

        # Plot the confusion matrix as a heatmap
        plt.figure(figsize=(8, 6))
        sns.set(font_scale=1.4)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                    xticklabels=labels, yticklabels=labels, linewidths=1, linecolor='black')

        # Add labels, title, and axis titles
        plt.xlabel('Predicted Labels', fontsize=14)
        plt.ylabel('True Labels', fontsize=14)
        plt.title('Confusion Matrix', fontsize=16)

        # Display the plot
        plt.show()
