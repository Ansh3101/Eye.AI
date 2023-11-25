using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Newtonsoft.Json;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Data;
using System.Windows.Documents;
using System.Windows.Input;
using System.Windows.Media;
using System.Windows.Media.Imaging;
using System.Net;
using System.Net.Http;
using System.Net.Http.Headers;
using System.Threading.Tasks;
using System.Windows.Shapes;
using System.IO;
using System.Windows.Media.Animation;

namespace EYE.AI
{
    /// <summary>
    /// Interaction logic for ShowPrediction.xaml
    /// </summary>
    /// 

    public partial class ShowPrediction : Window
    {

        private const string eyeDiseaseURL = "https://eyeai-eyedisease.herokuapp.com/";
        private const string cornealUlcersURL = "https://eyeai-cornealulcers.herokuapp.com/";
        private const string eyeBlindnessURL = "http://eyeai-blindness.herokuapp.com/";


        public ShowPrediction()
        {
            InitializeComponent();
        }

        private void Hyperlink_Click_1(object sender, RoutedEventArgs e)
        {
            this.Visibility = Visibility.Hidden;

            SelectionPage selectionPageInstance = new SelectionPage();

            selectionPageInstance.Visibility = Visibility.Visible;

        }

        private async void Window_Loaded(object sender, RoutedEventArgs e)
        {
            eyeImageOpener.Source = new BitmapImage(new Uri(PublicVariables.EyeImageURL));

            if (PublicVariables.EyeDiseaseAPI == true)
            {
                DoubleAnimation da = new DoubleAnimation
                {
                    From = 0,
                    To = 40,
                    Duration = new Duration(TimeSpan.FromSeconds(40)),
                    AutoReverse = false
                };

                greenDotLayover.BeginAnimation(OpacityProperty, da);

                string eyeDiseaseResult = await Task.Run(() => PostToEyeDiseaseAPI(PublicVariables.EyeImageURL));

                MessageBox.Show(eyeDiseaseResult); 
                PredictionProbability.Text = "0.9829";

            } else if (PublicVariables.CornealUlcersAPI == true)
            {

                DoubleAnimation da = new DoubleAnimation
                {
                    From = 0,
                    To = 40,
                    Duration = new Duration(TimeSpan.FromSeconds(40)),
                    AutoReverse = true
                };

                greenDotLayover.BeginAnimation(OpacityProperty, da);

                string cornealUlcersResult = await Task.Run(() => PostToCornealUlcersAPI(PublicVariables.EyeImageURL));

                MessageBox.Show(cornealUlcersResult);

                PredictionProbability.Text = "0.9829";

            } else if (PublicVariables.BlindnessAPI == true)
            {

                DoubleAnimation da = new DoubleAnimation
                {
                    From = 0,
                    To = 40,
                    Duration = new Duration(TimeSpan.FromSeconds(40)),
                    AutoReverse = true
                };

                greenDotLayover.BeginAnimation(OpacityProperty, da);

                string eyeBlindnessResult = await Task.Run(() => PostToBlindnessAPI(PublicVariables.EyeImageURL));

                MessageBox.Show(eyeBlindnessResult);

                PredictionProbability.Text = "0.9829";

            } else
            {
                MessageBox.Show("Please select the eye prediction!");
            }
        }

        private void Window_MouseDown(object sender, MouseButtonEventArgs e)
        {
            if (e.LeftButton == MouseButtonState.Pressed)
            {
                DragMove();
            }
        }

        
        //Clean up the code here
        public async Task<string> PostToEyeDiseaseAPI(string eyeImageModel)
        {
            var filePath = eyeImageModel;

            using (var multipartFormContent = new MultipartFormDataContent())
            {
                //Load the file and set the file's Content-Type header
                var fileStreamContent = new StreamContent(File.OpenRead(filePath));
                fileStreamContent.Headers.ContentType = new MediaTypeHeaderValue("image/jpg");

                //Add the file
                multipartFormContent.Add(fileStreamContent, name: "file", fileName: eyeImageModel);

                HttpClient httpClient = new HttpClient();       

                //Send it
                var response = await httpClient.PostAsync(eyeDiseaseURL, multipartFormContent);
                response.EnsureSuccessStatusCode();

                return await response.Content.ReadAsStringAsync();
            }
        }

        public async Task<string> PostToCornealUlcersAPI(string eyeImageModel)
        {
            var filePath = eyeImageModel;

            using (var multipartFormContent = new MultipartFormDataContent())
            {
                //Load the file and set the file's Content-Type header
                var fileStreamContent = new StreamContent(File.OpenRead(filePath));
                fileStreamContent.Headers.ContentType = new MediaTypeHeaderValue("image/jpg");

                //Add the file
                multipartFormContent.Add(fileStreamContent, name: "file", fileName: eyeImageModel);

                HttpClient httpClient = new HttpClient();

                //Send it
                var response = await httpClient.PostAsync(cornealUlcersURL, multipartFormContent);
                response.EnsureSuccessStatusCode();

                return await response.Content.ReadAsStringAsync();
            }
        }

        public async Task<string> PostToBlindnessAPI(string eyeImageModel)
        {
            var filePath = eyeImageModel;

            using (var multipartFormContent = new MultipartFormDataContent())
            {
                //Load the file and set the file's Content-Type header
                var fileStreamContent = new StreamContent(File.OpenRead(filePath));
                fileStreamContent.Headers.ContentType = new MediaTypeHeaderValue("image/jpg");

                //Add the file
                multipartFormContent.Add(fileStreamContent, name: "file", fileName: eyeImageModel);

                HttpClient httpClient = new HttpClient();

                //Send it
                var response = await httpClient.PostAsync(eyeBlindnessURL, multipartFormContent);
                response.EnsureSuccessStatusCode();

                return await response.Content.ReadAsStringAsync();
            }
        }
    }
}
 