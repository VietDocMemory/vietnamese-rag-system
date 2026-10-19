from src.generation.generator import RAGGenerator, normalize_answer


def test_normalize_answer_removes_presentation_only_text():
    answer = (
        "Theo NGỮ CẢNH, Giấy phép môi trường là văn bản do cơ quan có thẩm quyền cấp. "
        "[Nguồn 1]"
    )

    assert normalize_answer(answer) == (
        "Giấy phép môi trường là văn bản do cơ quan có thẩm quyền cấp."
    )


def test_normalize_answer_preserves_substantive_internal_text():
    answer = "Khái niệm được xác định theo tài liệu chuyên ngành và có hiệu lực."

    assert normalize_answer(answer) == answer


def test_normalize_answer_preserves_abstention():
    assert normalize_answer("Tài liệu không đề cập.") == "Tài liệu không đề cập."


def test_build_messages_formatting():
    generator = RAGGenerator(max_context_length=1000)
    query = "Luật bảo hiểm là gì?"
    contexts = [{"content": "Bảo hiểm là biện pháp chia sẻ rủi ro."}]

    messages = generator._build_messages(query, contexts)

    assert [message["role"] for message in messages] == ["system", "user"]
    assert (
        "chỉ sử dụng thông tin" in messages[0]["content"].lower()
    )

    assert "Luật bảo hiểm là gì?" in messages[1]["content"]
    assert "Bảo hiểm là biện pháp chia sẻ rủi ro." in messages[1]["content"]


def test_build_messages_truncation_limit():
    generator = RAGGenerator(max_context_length=150)  # tight limit
    query = "Hỏi điều khoản?"

    contexts = [
        {"content": "a" * 100},  # Length ~ 100 inside template
        {"content": "b" * 100},  # Should be truncated explicitly because length exceeds 150
    ]

    messages = generator._build_messages(query, contexts)
    user_content = messages[1]["content"]

    # First context block 'a' should be there
    assert "a" * 100 in user_content
    # Second context block 'b' should be dropped to respect context limit
    assert "b" * 100 not in user_content
