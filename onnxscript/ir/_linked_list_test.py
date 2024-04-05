from onnxscript.ir import _linked_list
import unittest
import parameterized


class _TestElement(_linked_list.Linkable):
    def __init__(self, value):
        self._prev = self
        self._next = self
        self._erased = False
        self._list = None
        self.value = value

    def __repr__(self) -> str:
        return f"_TestElement({self.value}, _prev={self._prev.value}, _next={self._next.value}, _erased={self._erased})"


class DoublyLinkedListTest(unittest.TestCase):
    def test_empty_list(self):
        linked_list = _linked_list.DoublyLinkedList(lambda: _TestElement(-1))
        self.assertEqual(len(linked_list), 0)
        self.assertEqual(list(linked_list), [])
        self.assertEqual(list(reversed(linked_list)), [])
        with self.assertRaises(IndexError):
            linked_list[0]
        with self.assertRaises(IndexError):
            linked_list[-1]

    def test_append_single_element(self):
        linked_list = _linked_list.DoublyLinkedList(lambda: _TestElement(-1))
        elem = _TestElement(0)
        linked_list.append(elem)

        self.assertEqual(len(linked_list), 1)
        self.assertEqual(linked_list[0], elem)
        self.assertEqual(linked_list[-1], elem)
        self.assertEqual(list(linked_list), [elem])
        self.assertEqual(list(reversed(linked_list)), [elem])
        with self.assertRaises(IndexError):
            linked_list[1]
        with self.assertRaises(IndexError):
            linked_list[-2]

    def test_append_multiple_elements(self):
        linked_list = _linked_list.DoublyLinkedList(lambda: _TestElement(-1))
        elems = [_TestElement(i) for i in range(3)]
        for elem in elems:
            linked_list.append(elem)

        self.assertEqual(len(linked_list), 3)
        self.assertEqual(linked_list[0], 0)
        self.assertEqual(linked_list[1], 1)
        self.assertEqual(linked_list[2], 2)
        self.assertEqual(linked_list[-1], 2)
        self.assertEqual(linked_list[-2], 1)
        self.assertEqual(linked_list[-3], 0)
        self.assertEqual(list(linked_list), elems)
        self.assertEqual(list(reversed(linked_list)), list(reversed(elems)))

    def test_extend(self):
        linked_list = _linked_list.DoublyLinkedList(lambda: _TestElement(-1))
        elems = [_TestElement(i) for i in range(3)]
        linked_list.extend(elems)

        self.assertEqual(len(linked_list), 3)
        self.assertEqual(linked_list[0], 0)
        self.assertEqual(linked_list[1], 1)
        self.assertEqual(linked_list[2], 2)
        self.assertEqual(linked_list[-1], 2)
        self.assertEqual(linked_list[-2], 1)
        self.assertEqual(linked_list[-3], 0)
        self.assertEqual(list(linked_list), elems)
        self.assertEqual(list(reversed(linked_list)), list(reversed(elems)))

    @parameterized.parameterized.expand(
        [
            ("single_element", [0], 0, [1], [0, 1]),
            ("single_element_negative_index", [0], -1, [1], [0, 1]),
            ("multiple_elements", [0], 0, [1, 2], [0, 1, 2]),
            ("multiple_elements_negative_index", [0], -1, [1, 2], [0, 1, 2]),
            (
                "multiple_original_elements_insert_at_start",
                [0, 1, 2],
                0,
                [42, 43],
                [0, 42, 43, 1, 2],
            ),
            (
                "multiple_original_elements_insert_at_middle",
                [0, 1, 2],
                1,
                [42, 43],
                [0, 1, 42, 43, 2],
            ),
            (
                "multiple_original_elements_insert_at_end",
                [0, 1, 2],
                1,
                [42, 43],
                [0, 1, 2, 42, 43],
            ),
        ]
    )
    def test_insert_after(
        self, _: str, original: list[int], location: int, insertion: list[int], expected: list
    ) -> None:
        # Construct the original list
        linked_list = _linked_list.DoublyLinkedList(lambda: _TestElement(-1))
        elems = [_TestElement(i) for i in original]
        linked_list.extend(elems)

        # Create the new elements
        new_elements = [_TestElement(i) for i in insertion]
        linked_list.insert_after(elems[location], new_elements)

        # Check the list
        self.assertEqual(len(linked_list), len(expected))
        self.assertEqual([elem.value for elem in linked_list], expected)


if __name__ == "__main__":
    unittest.main()
