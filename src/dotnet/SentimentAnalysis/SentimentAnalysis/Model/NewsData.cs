using Microsoft.ML.Runtime.Api;

namespace SentimentAnalysis.Model
{
    /// <summary>
    /// Represents the BBC news data.
    /// </summary>
    class NewsData
    {
        [Column(ordinal: "0")]
        public string Text;

        [Column(ordinal: "1", name: "Label")]
        public string Label;
    }
}
