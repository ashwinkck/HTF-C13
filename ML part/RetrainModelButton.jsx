import React, { useState } from 'react';

const RetrainModelButton = () => {
  const [status, setStatus] = useState("");

  const handleRetrain = async () => {
    try {
      const res = await fetch("/api/retrain", {
        method: "POST",
      });

      const result = await res.json();
      setStatus(result.message || "Model retrained!");
    } catch (err) {
      console.error(err);
      setStatus("Retrain failed.");
    }
  };

  return (
    <div className="mt-4">
      <button
        className="bg-green-500 hover:bg-green-600 text-white font-bold py-2 px-4 rounded"
        onClick={handleRetrain}
      >
        Retrain Model
      </button>
      {status && <p className="mt-2 text-sm">{status}</p>}
    </div>
  );
};

export default RetrainModelButton;
