/*
 * Copyright 2018 Jan Tschada
 * 
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 * 
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Models;
using Microsoft.ML.Trainers;
using Microsoft.ML.Transforms;
using SentimentAnalysis.Model;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Threading.Tasks;

/*
 * See: https://docs.microsoft.com/de-de/dotnet/machine-learning/tutorials/sentiment-analysis
 */

namespace SentimentAnalysis
{
    class Program
    {
        private const string _dataPath = @"wikipedia-detox-250-line-data.tsv";
        private const string _modelPath = @"wikipedia-detox-250-line-data.train";
        private const string _testDataPath = @"wikipedia-detox-250-line-test.tsv";

        static async Task<PredictionModel<SentimentData, SentimentPrediction>> Train()
        {
            var pipeline = new LearningPipeline();
            pipeline.Add(new TextLoader(_dataPath).CreateFrom<SentimentData>());
            pipeline.Add(new TextFeaturizer("Features", "SentimentText"));
            pipeline.Add(new FastTreeBinaryClassifier() { NumLeaves = 5, NumTrees = 5, MinDocumentsInLeafs = 2 });
            var model = pipeline.Train<SentimentData, SentimentPrediction>();
            await model.WriteAsync(_modelPath);
            return model;
        }

        static void Evaluate(PredictionModel<SentimentData, SentimentPrediction> model)
        {
            var testData = new TextLoader(_testDataPath).CreateFrom<SentimentData>();
            var evaluator = new BinaryClassificationEvaluator();
            var metrics = evaluator.Evaluate(model, testData);
            Console.WriteLine();
            Console.WriteLine("PredictionModel quality metrics evaluation");
            Console.WriteLine("------------------------------------------");
            Console.WriteLine($"Accuracy: {metrics.Accuracy:P2}");
            Console.WriteLine($"Auc: {metrics.Auc:P2}");
            Console.WriteLine($"F1Score: {metrics.F1Score:P2}");
        }

        static void Predict(PredictionModel<SentimentData, SentimentPrediction> model)
        {
            IEnumerable<SentimentData> sentiments = new[]
            {
                new SentimentData
                {
                    SentimentText = "Please refrain from adding nonsense to Wikipedia."
                },
                new SentimentData
                {
                    SentimentText = "He is the best, and the article should say that."
                }
            };

            IEnumerable<SentimentPrediction> predictions = model.Predict(sentiments);
            Console.WriteLine();
            Console.WriteLine("Sentiment Predictions");
            Console.WriteLine("---------------------");

            var sentimentsAndPredictions = sentiments.Zip(predictions, (sentiment, prediction) => (sentiment, prediction));
            foreach (var item in sentimentsAndPredictions)
            {
                Console.WriteLine($"Sentiment: {item.sentiment.SentimentText} | Prediction: {(item.prediction.Sentiment ? "Positive" : "Negative")}");
            }
            Console.WriteLine();
        }

        static IEnumerable<Open311Data> OpenFile(string path, int expectedTokenCount, int codeIndex, int textIndex)
        {
            var standardizer = new StopwordsStandardizer(@"german_stopwords_full.txt");
            var serviceTypes = new Open311ServiceTypes();
            var unknownTypes = new HashSet<string>();
            using (var reader = new StreamReader(path))
            {
                var header = reader.ReadLine();
                string line;
                const char Delimiter = '\t';
                while (null != (line = reader.ReadLine()))
                {
                    var tokens = line.Split(Delimiter);
                    var tokenCount = tokens.Length;
                    if (expectedTokenCount == tokenCount)
                    {
                        var record = new Open311Data();
                        var serviceType = tokens[codeIndex];
                        if (float.TryParse(serviceType, out float code))
                        {
                            // Validate the service type
                            if (serviceTypes.IsKnownServiceType(code))
                            {
                                record.Code = code;
                                var userRequest = tokens[textIndex];
                                record.Text = standardizer.Standardize(userRequest);
                                yield return record;
                            }
                            else
                            {
                                unknownTypes.Add(serviceType);
                            }
                        }
                    }
                }
            }

            if (0 < unknownTypes.Count)
            {
                Console.WriteLine($"{unknownTypes.Count} unknown service types!");
            }
        }

        static async Task<PredictionModel<Open311Data, Open311DataPrediction>> TrainOpen311(string dataPath)
        {
            var pipeline = new LearningPipeline();
            var dataSource = CollectionDataSource.Create(OpenFile(dataPath, 3, 0, 2));
            pipeline.Add(dataSource);
            pipeline.Add(new Dictionarizer(@"Label"));
            pipeline.Add(new TextFeaturizer(@"Features", @"Request"));
            pipeline.Add(new StochasticDualCoordinateAscentClassifier());
            pipeline.Add(new PredictedLabelColumnOriginalValueConverter { PredictedLabelColumn = @"PredictedLabel" });

            var model = pipeline.Train<Open311Data, Open311DataPrediction>();
            await model.WriteAsync(_modelPath);
            return model;
        }

        static void EvaluateOpen311(PredictionModel<Open311Data, Open311DataPrediction> model, string testDataPath)
        {
            var testData = CollectionDataSource.Create(OpenFile(testDataPath, 3, 0, 2));
            var evaluator = new ClassificationEvaluator();
            var metrics = evaluator.Evaluate(model, testData);
            Console.WriteLine();
            Console.WriteLine("PredictionModel quality metrics evaluation");
            Console.WriteLine("------------------------------------------");
            Console.WriteLine($"Accuracy Macro: {metrics.AccuracyMacro:P2}");
            Console.WriteLine($"Accuracy Micro: {metrics.AccuracyMicro:P2}");
            Console.WriteLine($"Top KAccuracy: {metrics.TopKAccuracy:P2}");
            Console.WriteLine($"LogLoss: {metrics.LogLoss:P2}");
            for (var classIndex = 0; classIndex < metrics.PerClassLogLoss.Length; classIndex++)
            {
                Console.WriteLine($"Class: {classIndex} - {metrics.PerClassLogLoss[classIndex]:P2}");
            }
        }

        static void PredictOpen311(PredictionModel<Open311Data, Open311DataPrediction> model)
        {
            IEnumerable<Open311Data> sentiments = new[]
            {
                new Open311Data
                {
                    Code = 9,
                    Text = @"Seit einigen Wochen steht dort ein abgemeldetes Fahrzeug (ehemals DHL) mit Kurzzeitkennzeichen (Mai 2015)"
                },
                new Open311Data
                {
                    Code = 2,
                    Text = @"Glassplitter am Straßenrand"
                },
                new Open311Data
                {
                    Code = 22,
                    Text = @"Ich habe bereits 2x telefonisch mitgeteilt, dass der Kanaldeckel in der Straßenmitte klappert, erstmals vor zwei Monaten, zuletzt vor drei Wochen. Der Deckel ist mit einem Kreuz markiert worden, sonst ist nichts passiert. Der Deckel verursacht viel Lärm, besonders störend in der Nacht. Der Deckel gefährdet zusätzlich die Verkehrssicherheit!"
                },
                new Open311Data
                {
                    Code = 26,
                    Text = @"wo auch der Grünabfall-Container oft bereit steht"
                },
                new Open311Data
                {
                    Code = 2,
                    Text = @"Grüne Schule wurde leider mutwillig beschädigt. Pflastersteine sind gelockert und werden immer wieder durch die Gegend geworfen. Je länger man nun wartet die Pflasterlücke zu schließen, desto aufwändiger wird es. Im Moment sollte es aber innerhalb von einer halben Stunde zu reparieren sein."
                },
                new Open311Data
                {
                    Code = 2,
                    Text = @"Am Spielplatz auf der Wiese an einem  Baum in der Nähe der Kleinkinderschaukel / gegenüber der Sprunggrube"
                },
                new Open311Data
                {
                    Code = 8,
                    Text = @"Bei uns ist schon wieder die Straßenlaterne defekt!"
                }
            };

            IEnumerable<Open311DataPrediction> predictions = model.Predict(sentiments);
            Console.WriteLine();
            Console.WriteLine("Open311 Predictions");
            Console.WriteLine("-------------------");

            var serviceTypes = new Open311ServiceTypes();
            var sentimentsAndPredictions = sentiments.Zip(predictions, (sentiment, prediction) => (sentiment, prediction));
            foreach (var item in sentimentsAndPredictions)
            {
                var text = item.sentiment.Text;
                var code = item.sentiment.Code;
                var serviceType = item.prediction.ServiceType;
                var serviceName = serviceTypes.IsKnownServiceType(serviceType) ? serviceTypes.GetNameFromServiceType(serviceType) : @"Unknown";
                Console.WriteLine($"Sentiment: {text}");
                Console.WriteLine($"Code: {code}\tPrediction: {serviceType} - {serviceName}");
                Console.WriteLine();
            }
            Console.WriteLine();
        }

        static async Task Main(string[] args)
        {
            //var model = await Train();
            //Evaluate(model);
            //Predict(model);

            var model = await TrainOpen311(@"D:\OpenData.Bonn\open311-ml-requests.tsv");
            EvaluateOpen311(model, @"D:\OpenData.Bonn\open311-ml-requests-evaluate.tsv");
            PredictOpen311(model);
        }
    }
}
