﻿/*
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
using Microsoft.ML.Runtime.Api;

namespace SentimentAnalysis.Model
{
    /// <summary>
    /// Represents the Open311 data.
    /// </summary>
    class Open311Data
    {
        [Column(ordinal: @"0", name: @"Type")]
        public float ServiceType { get; set; }

        [Column(ordinal: @"1", name: @"Request")]
        public string UserRequest { get; set; }
    }
}