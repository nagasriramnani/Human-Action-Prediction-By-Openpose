import React from 'react';
import { cn } from '@/lib/utils';

// Simple Progress Component
const Progress = React.forwardRef(({ className, value, ...props }, ref) => (
    <div
        ref={ref}
        className={cn("relative h-4 w-full overflow-hidden rounded-full bg-secondary", className)}
        {...props}
    >
        <div
            className="h-full w-full flex-1 bg-primary transition-all"
            style={{ transform: `translateX(-${100 - (value || 0)}%)` }}
        />
    </div>
));
Progress.displayName = "Progress";

const Card = React.forwardRef(({ className, ...props }, ref) => (
    <div ref={ref} className={cn("rounded-lg border bg-card text-card-foreground shadow-sm", className)} {...props} />
));
Card.displayName = "Card";

const CardHeader = React.forwardRef(({ className, ...props }, ref) => (
    <div ref={ref} className={cn("flex flex-col space-y-1.5 p-6", className)} {...props} />
));
CardHeader.displayName = "CardHeader";

const CardTitle = React.forwardRef(({ className, ...props }, ref) => (
    <h3 ref={ref} className={cn("text-2xl font-semibold leading-none tracking-tight", className)} {...props} />
));
CardTitle.displayName = "CardTitle";

const CardContent = React.forwardRef(({ className, ...props }, ref) => (
    <div ref={ref} className={cn("p-6 pt-0", className)} {...props} />
));
CardContent.displayName = "CardContent";

export default function Results({ results }) {
    if (!results) return null;

    const { classes, probabilities, top_class, top_probability } = results;

    if (!classes || !probabilities) return null;

    // Combine classes and probabilities and sort
    const data = classes.map((cls, idx) => ({
        cls,
        prob: probabilities[idx] * 100
    })).sort((a, b) => b.prob - a.prob);

    return (
        <Card className="w-full border-2 shadow-lg">
            <CardHeader>
                <CardTitle>Analysis Results</CardTitle>
            </CardHeader>
            <CardContent className="space-y-6">
                <div className="text-center">
                    <div className="text-sm text-muted-foreground">Top Prediction</div>
                    <div className="text-3xl font-bold text-primary capitalize">{top_class}</div>
                    <div className="text-sm text-muted-foreground">{(top_probability * 100).toFixed(1)}% Confidence</div>
                </div>

                {results.video_url && (
                    <div className="rounded-lg overflow-hidden border">
                        <video
                            src={`http://localhost:8000${results.video_url}`}
                            controls
                            autoPlay
                            loop
                            className="w-full"
                        />
                        <div className="p-2 text-xs text-center text-muted-foreground bg-muted">
                            Processed Video with Skeleton Overlay
                        </div>
                    </div>
                )}

                <div className="space-y-4">
                    {data.map((item) => (
                        <div key={item.cls} className="space-y-1">
                            <div className="flex justify-between text-sm">
                                <span className="capitalize font-medium">{item.cls}</span>
                                <span className="text-muted-foreground">{item.prob.toFixed(1)}%</span>
                            </div>
                            <Progress value={item.prob} />
                        </div>
                    ))}
                </div>
            </CardContent>
        </Card>
    );
}
