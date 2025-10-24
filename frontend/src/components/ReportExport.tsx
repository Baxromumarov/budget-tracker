import { useState } from "react";
import { downloadReport } from "../api/reports";

interface ReportExportProps {
  month: number;
  year: number;
}

export default function ReportExport({ month, year }: ReportExportProps) {
  const [loading, setLoading] = useState<"csv" | "json" | null>(null);

  const handleDownload = async (format: "csv" | "json") => {
    try {
      setLoading(format);
      const blob = await downloadReport(month, year, format);
      const url = window.URL.createObjectURL(blob);
      const link = document.createElement("a");
      link.href = url;
      link.setAttribute("download", `report_${year}_${month}.${format}`);
      document.body.appendChild(link);
      link.click();
      link.remove();
      window.URL.revokeObjectURL(url);
    } finally {
      setLoading(null);
    }
  };

  return (
    <div style={{ display: "flex", gap: "1rem" }}>
      <button className="btn" onClick={() => void handleDownload("csv")} disabled={loading === "csv"}>
        {loading === "csv" ? "Downloading..." : "Download CSV"}
      </button>
      <button className="btn secondary" onClick={() => void handleDownload("json")} disabled={loading === "json"}>
        {loading === "json" ? "Downloading..." : "Download JSON"}
      </button>
    </div>
  );
}
