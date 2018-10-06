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
using System.Collections.Generic;
using System.IO;

namespace SentimentAnalysis.Model
{
    /// <summary>
    /// Represents a stop word standardizer using german stopwords.
    /// </summary>
    class StopwordsStandardizer
    {
        internal StopwordsStandardizer(string filePath)
        {
            Stopwords = new HashSet<string>();
            using (var reader = new StreamReader(filePath))
            {
                string line;
                while (null != (line = reader.ReadLine()))
                {
                    if (!line.StartsWith(@";"))
                    {
                        Stopwords.Add(line);
                    }
                }
            }
        }

        private readonly HashSet<string> Stopwords;

        internal string Standardize(string text)
        {
            var tokens = text.Split(new[] { ' ', ',', ';', '-', '/', '(', ')', '%', '.', '?', '!' });
            foreach (var token in tokens)
            {
                if (Stopwords.Contains(token))
                {
                    text = text.Replace(token, string.Empty);
                }
            }
            return text;
        }
    }
}
