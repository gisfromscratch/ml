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
using System.Text;

namespace SentimentAnalysis.Model
{
    /// <summary>
    /// Represents a synonym standardizer for the german language.
    /// </summary>
    class SynonymStandardizer
    {
        internal SynonymStandardizer()
        {
            Synonyms = new Dictionary<string, string>();
            Synonyms.Add(@"Verkehrsampel", @"Ampel");

            Synonyms.Add(@"Glassplitter", @"Glas");
            Synonyms.Add(@"Glasscherbe", @"Glas");

            Synonyms.Add(@"Grünpate", @"Pate");

            Synonyms.Add(@"Grünüberwuchs", @"Überwuchs");
            Synonyms.Add(@"Verkehrsraum", @"Verkehr");
            Synonyms.Add(@"Straßenverkehr", @"Verkehr");

            Synonyms.Add(@"unüberwacht", @"herrenlos");
            Synonyms.Add(@"unbewacht", @"herrenlos");
            Synonyms.Add(@"ungesichert", @"herrenlos");
            Synonyms.Add(@"unbehütet", @"herrenlos");

            Synonyms.Add(@"Drahtesel", @"Fahrrad");
            Synonyms.Add(@"Vehikel", @"Fahrrad");
            Synonyms.Add(@"Zweirad", @"Fahrrad");
            Synonyms.Add(@"Rad", @"Fahrrad");
            Synonyms.Add(@"Fahrzeug", @"Auto");

            Synonyms.Add(@"Wagen", @"Auto");
            Synonyms.Add(@"Karre", @"Auto");
            Synonyms.Add(@"Kraftfahrzeug", @"Auto");
            Synonyms.Add(@"KFZ", @"Auto");
            Synonyms.Add(@"Personenkraftwagen", @"Auto");
            Synonyms.Add(@"PKW", @"Auto");
            Synonyms.Add(@"Verkehrsmittel", @"Auto");
            Synonyms.Add(@"Gefährt", @"Auto");
            Synonyms.Add(@"Schlitten", @"Auto");

            Synonyms.Add(@"Gerümpel", @"Schrott");
            Synonyms.Add(@"Schund", @"Schrott");
            Synonyms.Add(@"Ramsch", @"Schrott");
            Synonyms.Add(@"Kram", @"Schrott");

            Synonyms.Add(@"Straßenlaterne", @"Laterne");
            Synonyms.Add(@"Beleuchtung", @"Laterne");
            Synonyms.Add(@"Beleuchtungskörper", @"Laterne");
            Synonyms.Add(@"Straßenbeleuchtung", @"Laterne");
            Synonyms.Add(@"Lampe", @"Laterne");
            Synonyms.Add(@"Straßenlampe", @"Laterne");
            Synonyms.Add(@"Leuchte", @"Laterne");

            Synonyms.Add(@"Pfosten", @"Poller");
            Synonyms.Add(@"Pfeiler", @"Poller");
            Synonyms.Add(@"Pfahl", @"Poller");

            Synonyms.Add(@"Altpapier", @"Papier");
            Synonyms.Add(@"Papiercontainer", @"Papier");
            Synonyms.Add(@"Papiertonne", @"Papier");

            Synonyms.Add(@"Straßenkanaldeckel", @"Kanaldeckel");

            Synonyms.Add(@"Verkehrsschild", @"Straßenschild");
            Synonyms.Add(@"Verkehrszeichen", @"Straßenschild");

            Synonyms.Add(@"Müllkippe", @"Müll");
            Synonyms.Add(@"Müllabladeplatz", @"Müll");
            Synonyms.Add(@"Abladeplatz", @"Müll");
            Synonyms.Add(@"Deponie", @"Müll");
            Synonyms.Add(@"Müllhalde", @"Müll");
            Synonyms.Add(@"Abfall", @"Müll");
            Synonyms.Add(@"Abfallberg", @"Müll");
            Synonyms.Add(@"Abfallhaufen", @"Müll");
            Synonyms.Add(@"Sperrmüll", @"Müll");
        }

        private readonly IDictionary<string, string> Synonyms;

        internal string Standardize(string text)
        {
            var tokens = text.Split(new[] { ' ', ',', ';', '-', '/', '(', ')', '%', '.', '?', '!' });
            foreach (var token in tokens)
            {
                if (Synonyms.ContainsKey(token))
                {
                    text = text.Replace(token, Synonyms[token]);
                }
            }
            return text;
        }

        internal string Tag(string text)
        {
            var tagBuilder = new StringBuilder();
            var hasTags = false;
            var tokens = text.Split(new[] { ' ', ',', ';', '-', '/', '(', ')', '%', '.', '?', '!' });
            foreach (var token in tokens)
            {
                if (Synonyms.ContainsKey(token))
                {
                    if (hasTags)
                    {
                        tagBuilder.Append(@"|");
                    }
                    tagBuilder.Append(Synonyms[token]);
                    hasTags = true;
                }
            }
            return tagBuilder.ToString();
        }
    }
}
