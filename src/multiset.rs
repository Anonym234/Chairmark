use std::collections::BTreeMap;

#[derive(Debug, Clone)]
pub struct MultiSet<T: Ord> {
    map: BTreeMap<T, usize>,
    size: usize,
}

impl<T: Default + Ord> Default for MultiSet<T> {
    fn default() -> Self {
        Self {
            map: BTreeMap::default(),
            size: 0,
        }
    }
}

impl<T: Ord> MultiSet<T> {
    pub fn new() -> Self {
        Self {
            map: BTreeMap::new(),
            size: 0,
        }
    }

    pub fn size(&self) -> usize {
        self.size
    }

    pub fn is_empty(&self) -> bool {
        self.size() == 0
    }

    pub fn min(&self) -> Option<&T> {
        self.map.first_key_value().map(|(a, _)| a)
    }

    pub fn max(&self) -> Option<&T> {
        self.map.last_key_value().map(|(a, _)| a)
    }

    pub fn insert(&mut self, item: T) {
        if let Some(already) = self.map.get_mut(&item) {
            *already += 1;
        } else {
            self.map.insert(item, 1);
        }

        self.size += 1;
    }

    pub fn remove(&mut self, item: &T) -> bool {
        if let Some(count) = self.map.get_mut(item) {
            if *count > 0 {
                *count -= 1;
                if *count == 0 {
                    self.map.remove(item);
                }
                true
            } else {
                false
            }
        } else {
            false
        }
    }

    pub fn iter_counts(&self) -> std::collections::btree_map::Iter<T, usize> {
        (&self.map).into_iter()
    }
}

impl<T: Ord + Clone> MultiSet<T> {
    pub fn iter(&self) -> Iter<T> {
        self.into_iter()
    }
}

impl<'a, T: Ord + Clone> IntoIterator for &'a MultiSet<T> {
    type Item = T;
    type IntoIter = Iter<'a, T>;

    fn into_iter(self) -> Self::IntoIter {
        Iter::from(&self.map)
    }
}

#[derive(Debug, Clone)]
pub struct Iter<'a, T> {
    iter: <&'a BTreeMap<T, usize> as IntoIterator>::IntoIter,
    current: Option<T>,
    left: usize,
}

impl<'a, T> From<&'a BTreeMap<T, usize>> for Iter<'a, T> {
    fn from(value: &'a BTreeMap<T, usize>) -> Self {
        Self {
            iter: value.into_iter(),
            current: None,
            left: 0,
        }
    }
}

impl<'a, T: Ord + Clone> Iterator for Iter<'a, T> {
    type Item = T;

    fn next(&mut self) -> Option<Self::Item> {
        // no current item, get if available
        if self.left == 0 {
            let (item, count) = self
                .iter
                .next()
                .map(|(item, count)| (item.clone(), *count))
                .unzip();

            assert!(count.is_none() || count.unwrap() > 0);

            self.current = item;
            self.left = count.unwrap_or(0);
        }

        // return current item, if there
        match self.left {
            0 => None,
            1 => self.current.take(),
            _ => {
                let Some(current) = &self.current else {
                    unreachable!()
                };

                self.left -= 1;
                Some(current.clone())
            }
        }
    }
}

impl<T: Ord + Clone> IntoIterator for MultiSet<T> {
    type Item = T;
    type IntoIter = IntoIter<T>;

    fn into_iter(self) -> Self::IntoIter {
        IntoIter::from(self)
    }
}

#[derive(Debug)]
pub struct IntoIter<T> {
    iter: <BTreeMap<T, usize> as IntoIterator>::IntoIter,
    current: Option<T>,
    left: usize,
}

impl<T: Ord> From<MultiSet<T>> for IntoIter<T> {
    fn from(value: MultiSet<T>) -> Self {
        Self {
            iter: value.map.into_iter(),
            current: None,
            left: 0,
        }
    }
}

impl<T: Ord + Clone> Iterator for IntoIter<T> {
    type Item = T;

    fn next(&mut self) -> Option<Self::Item> {
        // no current item, get if available
        if self.left == 0 {
            let (item, count) = self.iter.next().map(|(item, count)| (item, count)).unzip();

            assert!(count.is_none() || count.unwrap() > 0);

            self.current = item;
            self.left = count.unwrap_or(0);
        }

        assert!(self.left == 0 || self.current.is_some());

        // return current item, if there
        match self.left {
            0 => None,
            1 => self.current.take(),
            _ => {
                let Some(current) = &self.current else {
                    unreachable!()
                };

                self.left -= 1;
                Some(current.clone())
            }
        }
    }
}
