
#!/usr/bin/env python3
"""Return all students sorted by average score."""


def top_students(mongo_collection):
    """Return students ordered by average score, highest first."""
    students = []

    for student in mongo_collection.find():
        topics = student.get("topics", [])

        if topics:
            average_score = sum(
                topic.get("score", 0) for topic in topics
            ) / len(topics)
        else:
            average_score = 0

        student["averageScore"] = average_score
        students.append(student)

    return sorted(
        students,
        key=lambda student: student["averageScore"],
        reverse=True,
    )
