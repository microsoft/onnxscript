import unittest

import parameterized

from onnxscript.ir import _linked_list


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
        self.assertEqual(linked_list[0], elems[0])
        self.assertEqual(linked_list[1], elems[1])
        self.assertEqual(linked_list[2], elems[2])
        self.assertEqual(linked_list[-1], elems[2])
        self.assertEqual(linked_list[-2], elems[1])
        self.assertEqual(linked_list[-3], elems[0])
        self.assertEqual(list(linked_list), elems)
        self.assertEqual(list(reversed(linked_list)), list(reversed(elems)))

    def test_extend(self):
        linked_list = _linked_list.DoublyLinkedList(lambda: _TestElement(-1))
        elems = [_TestElement(i) for i in range(3)]
        linked_list.extend(elems)

        self.assertEqual(len(linked_list), 3)
        self.assertEqual(linked_list[0], elems[0])
        self.assertEqual(linked_list[1], elems[1])
        self.assertEqual(linked_list[2], elems[2])
        self.assertEqual(linked_list[-1], elems[2])
        self.assertEqual(linked_list[-2], elems[1])
        self.assertEqual(linked_list[-3], elems[0])
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
                2,
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

    @parameterized.parameterized.expand(
        [
            ("single_element", [0], 0, [1], [1, 0]),
            ("single_element_negative_index", [0], -1, [1], [1, 0]),
            ("multiple_elements", [0], 0, [1, 3], [1, 3, 0]),
            ("multiple_elements_negative_index", [0], -1, [1, 3], [1, 3, 0]),
            (
                "multiple_original_elements_insert_at_start",
                [0, 1, 2],
                0,
                [42, 43],
                [42, 43, 0, 1, 2],
            ),
            (
                "multiple_original_elements_insert_at_middle",
                [0, 1, 2],
                1,
                [42, 43],
                [0, 42, 43, 1, 2],
            ),
            (
                "multiple_original_elements_insert_at_end",
                [0, 1, 2],
                2,
                [42, 43],
                [0, 1, 42, 43, 2],
            ),
        ]
    )
    def test_insert_before(
        self, _: str, original: list[int], location: int, insertion: list[int], expected: list
    ) -> None:
        # Construct the original list
        linked_list = _linked_list.DoublyLinkedList(lambda: _TestElement(-1))
        elems = [_TestElement(i) for i in original]
        linked_list.extend(elems)

        # Create the new elements
        new_elements = [_TestElement(i) for i in insertion]
        linked_list.insert_before(elems[location], new_elements)

        # Check the list
        self.assertEqual(len(linked_list), len(expected))
        self.assertEqual([elem.value for elem in linked_list], expected)
        self.assertEqual([elem.value for elem in reversed(linked_list)], expected[::-1])

    @parameterized.parameterized.expand(
        [
            ("start", 0, [1, 2]),
            ("middle", 1, [0, 2]),
            ("end", 2, [0, 1]),
            ("start_negative", -1, [0, 1]),
            ("middle_negative", -2, [0, 2]),
            ("end_negative", -3, [1, 2]),
        ]
    )
    def test_remove(self, _: str, index: int, expected: list[int]) -> None:
        linked_list = _linked_list.DoublyLinkedList(lambda: _TestElement(-1))
        elems = [_TestElement(i) for i in range(3)]
        linked_list.extend(elems)

        linked_list.remove(elems[index])

        self.assertEqual(len(linked_list), 2)
        self.assertEqual([elem.value for elem in linked_list], expected)
        self.assertEqual([elem.value for elem in reversed(linked_list)], expected[::-1])

    def test_remove_raises_when_element_not_found(self) -> None:
        linked_list = _linked_list.DoublyLinkedList(lambda: _TestElement(-1))
        elems = [_TestElement(i) for i in range(3)]
        linked_list.extend(elems)

        with self.assertRaises(ValueError):
            linked_list.remove(_TestElement(3))

    def test_remove_raises_when_element_is_already_removed(self) -> None:
        linked_list = _linked_list.DoublyLinkedList(lambda: _TestElement(-1))
        elem = _TestElement(0)
        linked_list.append(elem)
        linked_list.remove(elem)

        with self.assertRaises(ValueError):
            linked_list.remove(elem)

    def test_append_self_does_nothing(self) -> None:
        linked_list = _linked_list.DoublyLinkedList(lambda: _TestElement(-1))
        elem = _TestElement(0)
        linked_list.append(elem)

        linked_list.append(elem)

        self.assertEqual(len(linked_list), 1)
        self.assertEqual(linked_list[0], elem)
        self.assertEqual(list(linked_list), [elem])
        self.assertEqual(list(reversed(linked_list)), [elem])

    def test_append_supports_appending_element_from_the_same_list(self) -> None:
        linked_list = _linked_list.DoublyLinkedList(lambda: _TestElement(-1))
        elems = [_TestElement(i) for i in range(3)]
        linked_list.extend(elems)

        linked_list.append(elems[1])

        self.assertEqual(len(linked_list), 3)
        self.assertEqual([elem.value for elem in linked_list], [0, 2, 1])
        self.assertEqual([elem.value for elem in reversed(linked_list)], [1, 2, 0])

    def test_extend_supports_extending_elements_from_the_same_list(self) -> None:
        linked_list = _linked_list.DoublyLinkedList(lambda: _TestElement(-1))
        elems = [_TestElement(i) for i in range(3)]
        linked_list.extend(elems)

        linked_list.extend(elems[::-1])

        self.assertEqual(len(linked_list), 3)
        self.assertEqual([elem.value for elem in linked_list], [2, 1, 0])
        self.assertEqual([elem.value for elem in reversed(linked_list)], [0, 1, 2])

    def test_insert_after_supports_inserting_element_from_the_same_list(self) -> None:
        linked_list = _linked_list.DoublyLinkedList(lambda: _TestElement(-1))
        elems = [_TestElement(i) for i in range(3)]
        linked_list.extend(elems)

        linked_list.insert_after(elems[0], [elems[2]])

        self.assertEqual(len(linked_list), 3)
        self.assertEqual([elem.value for elem in linked_list], [0, 2, 1])

    def test_insert_before_supports_inserting_element_from_the_same_list(self) -> None:
        linked_list = _linked_list.DoublyLinkedList(lambda: _TestElement(-1))
        elems = [_TestElement(i) for i in range(3)]
        linked_list.extend(elems)

        linked_list.insert_before(elems[0], [elems[2]])

        self.assertEqual(len(linked_list), 3)
        self.assertEqual([elem.value for elem in linked_list], [2, 0, 1])

    def test_iterator_supports_mutation_during_iteration_current_element(self) -> None:
        linked_list = _linked_list.DoublyLinkedList(lambda: _TestElement(-1))
        elems = [_TestElement(i) for i in range(3)]
        linked_list.extend(elems)

        for elem in linked_list:
            if elem.value == 1:
                linked_list.remove(elem)

        self.assertEqual(len(linked_list), 2)
        self.assertEqual([elem.value for elem in linked_list], [0, 2])
        self.assertEqual([elem.value for elem in reversed(linked_list)], [2, 0])

    def test_iterator_supports_mutation_during_iteration_previous_element(self) -> None:
        linked_list = _linked_list.DoublyLinkedList(lambda: _TestElement(-1))
        elems = [_TestElement(i) for i in range(3)]
        linked_list.extend(elems)

        for elem in linked_list:
            if elem.value == 1:
                linked_list.remove(elem)
                linked_list.remove(elems[0])

        self.assertEqual(len(linked_list), 1)
        self.assertEqual([elem.value for elem in linked_list], [2])
        self.assertEqual([elem.value for elem in reversed(linked_list)], [2])

    def test_iterator_supports_mutation_during_iteration_next_element(self) -> None:
        linked_list = _linked_list.DoublyLinkedList(lambda: _TestElement(-1))
        elems = [_TestElement(i) for i in range(3)]
        linked_list.extend(elems)

        for elem in linked_list:
            if elem.value == 1:
                linked_list.remove(elems[2])
                linked_list.remove(elem)

        self.assertEqual(len(linked_list), 1)
        self.assertEqual([elem.value for elem in linked_list], [0])
        self.assertEqual([elem.value for elem in reversed(linked_list)], [0])

    def test_iterator_supports_mutation_in_nested_iteration_right_of_iterator(self) -> None:
        linked_list = _linked_list.DoublyLinkedList(lambda: _TestElement(-1))
        elems = [_TestElement(i) for i in range(3)]
        linked_list.extend(elems)

        iter1_visited = []
        iter2_visited = []
        for elem in linked_list:
            iter1_visited.append(elem.value)
            for elem2 in linked_list:
                iter2_visited.append(elem2.value)
                if elem2.value == 1:
                    linked_list.remove(elem2)

        self.assertEqual(len(linked_list), 2)
        self.assertEqual(iter1_visited, [0, 2])
        self.assertEqual(iter2_visited, [0, 1, 2, 0, 2])
        self.assertEqual([elem.value for elem in linked_list], [0, 2])
        self.assertEqual([elem.value for elem in reversed(linked_list)], [2, 0])

    def test_iterator_supports_mutation_in_nested_iteration_when_iter_is_self(self) -> None:
        linked_list = _linked_list.DoublyLinkedList(lambda: _TestElement(-1))
        elems = [_TestElement(i) for i in range(3)]
        linked_list.extend(elems)

        iter1_visited = []
        iter2_visited = []
        for elem in linked_list:
            iter1_visited.append(elem.value)
            for elem2 in linked_list:
                iter2_visited.append(elem2.value)
                if elem2.value == 0:  # Remove the element the current iterator points to
                    linked_list.remove(elem2)

        self.assertEqual(len(linked_list), 2)
        self.assertEqual(iter1_visited, [0, 1, 2])
        self.assertEqual(iter2_visited, [0, 1, 2, 1, 2, 1, 2])
        self.assertEqual([elem.value for elem in linked_list], [1, 2])
        self.assertEqual([elem.value for elem in reversed(linked_list)], [2, 1])

    def test_iterator_supports_mutation_in_nested_iteration_left_of_iterator(self) -> None:
        linked_list = _linked_list.DoublyLinkedList(lambda: _TestElement(-1))
        elems = [_TestElement(i) for i in range(3)]
        linked_list.extend(elems)

        iter1_visited = []
        iter2_visited = []
        for elem in linked_list:
            iter1_visited.append(elem.value)
            for elem2 in linked_list:
                iter2_visited.append(elem2.value)
                if (
                    elem.value == 1 and elem2.value == 0
                ):  # Remove the element before the current iterator points to
                    linked_list.remove(elems[0])

        self.assertEqual(len(linked_list), 2)
        self.assertEqual(iter1_visited, [0, 1, 2])
        self.assertEqual(iter2_visited, [0, 1, 2, 0, 1, 2, 1, 2])
        self.assertEqual([elem.value for elem in linked_list], [1, 2])
        self.assertEqual([elem.value for elem in reversed(linked_list)], [2, 1])


    def test_insert_after_supports_element_from_different_list_during_iteration(self) -> None:
        linked_list = _linked_list.DoublyLinkedList(lambda: _TestElement(-1))
        elems = [_TestElement(i) for i in range(3)]
        linked_list.extend(elems)

        other_linked_list = _linked_list.DoublyLinkedList(lambda: _TestElement(-1))
        other_elem = _TestElement(42)
        other_linked_list.append(other_elem)

        for elem in linked_list:
            if elem.value == 1:
                linked_list.insert_after(elem, [other_elem])

        self.assertEqual(len(linked_list), 4)
        self.assertEqual([elem.value for elem in linked_list], [0, 1, 42, 2])
        self.assertEqual([elem.value for elem in reversed(linked_list)], [2, 42, 1, 0])
        self.assertEqual(len(other_linked_list), 0)
        self.assertEqual([elem.value for elem in other_linked_list], [])

    def test_insert_after_supports_taking_elements_from_another_doubly_linked_list(self) -> None:
        linked_list = _linked_list.DoublyLinkedList(lambda: _TestElement(-1))
        elems = [_TestElement(i) for i in range(3)]
        linked_list.extend(elems)

        other_linked_list = _linked_list.DoublyLinkedList(lambda: _TestElement(-1))
        other_elem = _TestElement(42)
        other_linked_list.append(other_elem)

        for elem in linked_list:
            if elem.value == 1:
                # This causes infinite loop if the implementation is incorrect
                linked_list.insert_after(elem, other_linked_list)

        self.assertEqual(len(linked_list), 4)
        self.assertEqual([elem.value for elem in linked_list], [0, 1, 42, 2])
        self.assertEqual([elem.value for elem in reversed(linked_list)], [2, 42, 1, 0])
        self.assertEqual(len(other_linked_list), 0)
        self.assertEqual([elem.value for elem in other_linked_list], [])

if __name__ == "__main__":
    unittest.main()
