using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Text.RegularExpressions;
using OpenCvSharp;
using OpenCvSharp.Dnn;

namespace OpenCvDnnVggFace
{
    class Program
    {
        static void Main()
        {
            //model can get from http://www.robots.ox.ac.uk/~vgg/software/vgg_face/
            var file = "jeremy-clarkson-v2.jpg";
            //var file = "john-cena.jpg";
            var prototxt = "VGG_FACE_deploy.prototxt";
            var model = "VGG_FACE.caffemodel";
            var labeltxt = "names.txt";

            //read all names
            var labels = ReadLabels(labeltxt);
            var org = Cv2.ImRead(file);
            var blob = CvDnn.BlobFromImage(org, 1, new Size(224, 224));
            var net = CvDnn.ReadNetFromCaffe(prototxt, model);
            net.SetInput(blob, "data");

            Stopwatch sw = new Stopwatch();
            sw.Start();
            //forward model
            var prob = net.Forward("prob");
            sw.Stop();
            Console.WriteLine($"Runtime:{sw.ElapsedMilliseconds} ms");

            //convert result to list
            var probList = new Dictionary<int,float>();
            for (int i = 0; i < prob.Width; i++)
            {
                probList.Add(i,prob.At<float>(0,i));
            }
            
            //get top 3
            var top3 = probList.OrderByDescending(x => x.Value).Take(3).ToList();
            foreach (var result in top3)
            {
                Console.WriteLine($"{labels[result.Key]}:{result.Value*100:0.00}%");
            }

            //draw result
            org.PutText($"{labels[top3.First().Key]}:{top3.First().Value * 100:0.00}%",
                new Point(0, 25), HersheyFonts.HersheyTriplex, 1, Scalar.OrangeRed);
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
            string patten = "(?<name>.*)\n";
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
                foreach (Match match in matches)
                {
                    var name = match.Groups[1].Value;

                    result.Add(name);
                }
                return result.ToArray();
            }
        }
    }
}
