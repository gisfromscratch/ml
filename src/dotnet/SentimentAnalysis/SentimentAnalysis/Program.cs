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

        static void TrainOpen311()
        {
            using (var reader = new StreamReader(@"D:\OpenData.Bonn\open311-requests.tsv"))
            {
                var header = reader.ReadLine();
                string line;
                const char Delimiter = '\t';
                while (null != (line = reader.ReadLine()))
                {
                    var tokens = line.Split(Delimiter);
                    var tokenCount = tokens.Length;
                    for(var tokenIndex = 0; tokenIndex < tokenCount; tokenIndex++)
                    {
                        var token = tokens[tokenIndex];
                    }
                }
            }
            /*
             * Train using in-memory data
            CollectionDataSource.Create<Open311Data>();
            */
        }

        static async Task Main(string[] args)
        {
            var model = await Train();
            Evaluate(model);
            Predict(model);
        }
    }
}
