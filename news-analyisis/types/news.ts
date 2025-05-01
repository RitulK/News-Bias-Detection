export interface NewsSource {
    _id: string;
    source_name: string;
    alignment_counts: {
      Left: number;
      Center: number;
      Right: number;
    };
    last_updated: string;
    total_articles: number;
    source_id?: string;
  }