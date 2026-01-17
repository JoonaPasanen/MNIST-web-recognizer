const handleBackgroundColor = (probability: number, numberGuess: number[]) => {
  if (probability == Math.max(...numberGuess)) return "green";
  if (probability > 0.05) return "gold";
};

const handleFontWeight = (probability: number, numberGuess: number[]) => {
  if (probability == Math.max(...numberGuess)) return "bold";
};

const handleScale = (probability: number, numberGuess: number[]) => {
  if (probability == Math.max(...numberGuess)) return "scale(1.05)";
};

interface Props {
  numberGuess: number[];
}

const ProbabilityBar = ({ numberGuess }: Props) => {
  return (
    <>
      {[...Array(10).keys()].map(function (i) {
        return (
          <div
            key={i}
            style={{
              backgroundColor: handleBackgroundColor(
                numberGuess[i],
                numberGuess
              ),
              fontWeight: handleFontWeight(numberGuess[i], numberGuess),
              transform: handleScale(numberGuess[i], numberGuess),
              color: "black",
              width: "400px",
              border: "1px solid darkgray",
              borderRadius: "16px",
              margin: "5px auto",
              transition: "all 0.3s ease",
            }}
          >
            <p>
              Probability for {i}:{" "}
              {numberGuess[i] ? (numberGuess[i] * 100).toFixed(2) : "?"}
            </p>
          </div>
        );
      })}
    </>
  );
};

export { ProbabilityBar };
