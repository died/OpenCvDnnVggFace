using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Text.RegularExpressions;
using OpenCvSharp;
using OpenCvSharp.Dnn;

namespace VggFaceCrop
{
    class Program
    {
        static void Main()
        {
            var file = "best-supporting-actress.jpg";
            //var file = "oscars-2017.jpg";
            var prototxt = "VGG_FACE_deploy.prototxt";
            var model = "VGG_FACE.caffemodel";
            var labeltxt = "names.txt";
            var cascade = "haarcascade_frontalface_default.xml";
            var org = Cv2.ImRead(file);

            //get face using haarcascades , https://github.com/opencv/opencv/tree/master/data/haarcascades
            var faceCascade = new CascadeClassifier();
            faceCascade.Load(cascade);
            var faces = faceCascade.DetectMultiScale(org, 1.1, 6, HaarDetectionType.DoRoughSearch, new Size(60, 60));
            var faceList = new List<Mat>();
            foreach (var rect in faces)
            {
                Cv2.Rectangle(org,rect,Scalar.Red);
                faceList.Add(org[rect]);
            }

            //read all names
            var labels = ReadLabels(labeltxt);
            var blob = CvDnn.BlobFromImages(faceList, 1, new Size(224, 224));
            var net = CvDnn.ReadNetFromCaffe(prototxt, model);
            net.SetInput(blob, "data");

            Stopwatch sw = new Stopwatch();
            sw.Start();
            //forward model
            var prob = net.Forward("prob");
            sw.Stop();
            Console.WriteLine($"Runtime:{sw.ElapsedMilliseconds} ms");

            for (int n = 0; n < prob.Height; n++)
            {
                //convert result to list
                var probList = new Dictionary<int, float>();
                for (int i = 0; i < prob.Width; i++)
                {
                    probList.Add(i, prob.At<float>(n, i));
                }

                //get top 1
                var top1 = probList.OrderByDescending(x => x.Value).First();
                var label = $"{labels[top1.Key]}:{top1.Value * 100:0.00}%";
                Console.WriteLine(label);

                //show if confidence > 50%
                if (top1.Value > 0.5)
                {
                    var textsize = Cv2.GetTextSize(label, HersheyFonts.HersheyTriplex, 0.5, 1, out var baseline);
                    var y = faces[n].TopLeft.Y - textsize.Height - baseline <= 0
                        ? faces[n].BottomRight.Y + textsize.Height + baseline : faces[n].TopLeft.Y - baseline;
                    //draw result
                    org.PutText(label, new Point(faces[n].TopLeft.X,y), HersheyFonts.HersheyTriplex, 0.5, Scalar.OrangeRed);
                }
            }
            using (new Window("image", org))
            {
                Cv2.WaitKey();
            }
        }

        /// <summary>
        /// Load label name list
        /// </summary>
        /// <param name="file"></param>
        /// <returns></returns>
        public static string[] ReadLabels(string file)
        {
            const string patten = "(?<name>.*)\n";
            var result = new List<string>();
            using (FileStream stream = File.OpenRead(file))
            using (StreamReader reader = new StreamReader(stream))
            {
                string text = reader.ReadToEnd();
                if (string.IsNullOrWhiteSpace(text))
                {
                    return result.ToArray();
                }
                Regex regex = new Regex(patten);
                var matches = regex.Matches(text);
                result.AddRange(from Match match in matches select match.Groups[1].Value);
                return result.ToArray();
            }
        }
    }
}
