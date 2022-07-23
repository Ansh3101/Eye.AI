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
using System.Windows.Threading;

namespace EYE.AI
{
    /// <summary>
    /// Interaction logic for LandingPage.xaml
    /// </summary>
    public partial class LandingPage : Window
    {

        private int increment = 0;
        public LandingPage()
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

        private void Window_Loaded(object sender, RoutedEventArgs e)
        {
            DispatcherTimer dispatcherTimer = new DispatcherTimer();

            dispatcherTimer.Interval = TimeSpan.FromSeconds(1);

            dispatcherTimer.Tick += dispatcherTimerTick;

            dispatcherTimer.Start();

        }

        private void dispatcherTimerTick(object sender, EventArgs e)
        {
            increment = increment + 1;

            //Hardcoded, could have been automated but progress bar only lasts for like 15 secs.
            if (increment == 3)
            {
                diseaseContent.Content = "Uveitis";
            }
            else if (increment == 6)
            {
                diseaseContent.Content = "Cataracts";
            }
            else if (increment == 9)
            {
                diseaseContent.Content = "Blindness";
            }
            else if (increment == 12)
            {
                diseaseContent.Content = "Corneal Ulcers";
            }

            if (increment == 15)
            {
                this.Visibility = Visibility.Hidden;

                SelectionPage selectionPageInstance = new SelectionPage();

                selectionPageInstance.Visibility = Visibility.Visible;
            }
        }
    }
}
