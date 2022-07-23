using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Data;
using System.Windows.Documents;
using System.Windows.Input;
using System.Windows.Media;
using System.Windows.Media.Imaging;
using System.Windows.Shapes;

namespace EYE.AI
{
    /// <summary>
    /// Interaction logic for SelectionPage.xaml
    /// </summary>
    public partial class SelectionPage : Window
    {
        public SelectionPage()
        {
            InitializeComponent();
        }

        private void Window_MouseDown(object sender, MouseButtonEventArgs e)
        {
            if (e.LeftButton == MouseButtonState.Pressed)
            {
                DragMove();
            }
        }

        private void Blindness_Click(object sender, RoutedEventArgs e)
        {
            this.Visibility = Visibility.Hidden;

            PredictionPage predictionPageInstance = new PredictionPage();
            predictionPageInstance.Visibility = Visibility.Visible;
        }

        private void EyeDiseases_Click(object sender, RoutedEventArgs e)
        {
            this.Visibility = Visibility.Hidden;

            PredictionPage predictionPageInstance = new PredictionPage();
            predictionPageInstance.Visibility = Visibility.Visible;
        }

        private void CornealUlcers_Click(object sender, RoutedEventArgs e)
        {
            this.Visibility = Visibility.Hidden;

            PredictionPage predictionPageInstance = new PredictionPage();
            predictionPageInstance.Visibility = Visibility.Visible;
        }
    }
}
