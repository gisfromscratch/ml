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

namespace SentimentAnalysis.Model
{
    /// <summary>
    /// Represents all service types from the city of Bonn.
    /// </summary>
    class Open311ServiceTypes
    {
        internal Open311ServiceTypes()
        {
            //TODO: Investigate why 21 is used twice!
            ServiceTypes = new Dictionary<float, string>();
            ServiceTypes.Add(1, @"Ampel defekt (Taste/Licht)");
            ServiceTypes.Add(2, @"Glassplitter");
            //ServiceTypes.Add(21, @"Graffiti");
            ServiceTypes.Add(5, @"Grünpate werden");
            ServiceTypes.Add(6, @"Grünüberwuchs Verkehrsraum");
            //ServiceTypes.Add(21, @"Gully/ Bachablauf verstopft");
            ServiceTypes.Add(9, @"Herrenlose Fahrräder, Fahrzeuge (Schrott)");
            ServiceTypes.Add(8, @"Laterne defekt");
            ServiceTypes.Add(24, @"Poller umgefahren");
            ServiceTypes.Add(25, @"Sammelcontainer Altpapier voll");
            ServiceTypes.Add(26, @"Sammelcontainer Grünschnitt voll");
            ServiceTypes.Add(22, @"Straßenkanaldeckel defekt");
            ServiceTypes.Add(23, @"Straßenschild defekt");
            ServiceTypes.Add(10, @"Wilde Müllkippe, Sperrmüllreste");
        }

        private IDictionary<float, string> ServiceTypes { get; }

        internal bool IsKnownServiceType(float serviceType)
        {
            return ServiceTypes.ContainsKey(serviceType);
        }

        internal string GetNameFromServiceType(float serviceType)
        {
            return ServiceTypes[serviceType];
        }
    }
}
