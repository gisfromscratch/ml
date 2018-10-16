using Microsoft.ML.Runtime.Api;

namespace SentimentAnalysis.Model
{
    /// <summary>
    /// Represents the news prediction.
    /// </summary>
    class NewsPrediction
    {
        [ColumnName("Score")]
        public float[] Score;
    }
}
