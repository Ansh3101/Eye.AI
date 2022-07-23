using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.IO;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Data;
using System.Windows.Documents;
using System.Windows.Input;
using Microsoft.Win32;
using System.Windows.Media;
using System.Windows.Media.Imaging;
using System.Windows.Shapes;

namespace EYE.AI
{
    /// <summary>
    /// Interaction logic for PredictionPage.xaml
    /// </summary>
    /// 

    
    public partial class PredictionPage : Window
    {

        public string eyeImageURL { get; set; }

        public BitmapImage eyeImage;

        public PredictionPage()
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

        private void Upload_Image_Click(object sender, RoutedEventArgs e)
        {
            OpenFileDialog openFileDialog = new OpenFileDialog();

            if (openFileDialog.ShowDialog() == true)
            {
                try
                {
                    eyeImageURL = openFileDialog.FileName;

                    MessageBox.Show(eyeImageURL);

                } catch(Exception exp)
                {
                    Console.WriteLine(exp.StackTrace);
                }
            }
        }
    }
}
