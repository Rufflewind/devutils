{-# LANGUAGE NoMonomorphismRestriction #-}
{-# LANGUAGE FlexibleInstances #-}
module Makegen
  ( module Makegen
  , mempty
  , (<>)
  ) where
import Control.Arrow
import Control.Monad
import Data.Dynamic
import Data.Maybe
import Data.Monoid
import Control.Monad.Trans.Writer.Strict
import Data.Map.Strict (Map)
import Data.Sequence (Seq)
import Data.Set (Set)
import qualified Data.Map.Strict as Map

-- | Equivalent to @Control.Lens.At.at@ for 'Map'.
at_Map :: (Functor f, Ord k) =>
          k -> (Maybe a -> f (Maybe a)) -> Map k a -> f (Map k a)
at_Map k f m = set <$> f (Map.lookup k m)
  where
    set Nothing  = Map.delete k m
    set (Just x) = Map.insert k x m

newtype VarMap
  = VarMap (Map TypeRep Dynamic)
  deriving Show

-- | Union two 'Map's.  If the 'Map's contain conflicting elements, the list
-- of conflicting elements is returned via 'Left'.
unionEq_Map :: (Ord k, Eq a) =>
               Map k a
            -> Map k a
            -> Either [(k, a, a)] (Map k a)
unionEq_Map m1 m2
  | null conflicts = Right (m1 <> m2)
  | otherwise      = Left conflicts
  where
    conflicts = do
      (k, v1) <- Map.toAscList (Map.intersection m1 m2)
      let v2 = m2 Map.! k
      guard (v1 /= v2)
      pure (k, v1, v2)

field :: (Functor f, Eq a, Monoid a, Typeable a) =>
         (a -> f a) -> VarMap -> f VarMap
field f (VarMap m) =
  case undefined of
    dummy_a ->
      let set x | x == mempty = Nothing
                | otherwise   = Just (toDyn (x `asTypeOf` dummy_a))
          get y = fromMaybe mempty (fromDynamic =<< y)
          upd y = set <$> f (get y)
      in VarMap <$> at_Map (typeOf dummy_a) upd m

data Makefile =
  Makefile
  { _rules :: Map String ([String], [String])
  , _macros :: Map String String
  }

data MakefileError
  = ERuleConflict (String, ([String], [String]), ([String], [String]))
  | EMacroConflict (String, String, String)
  deriving (Eq, Ord, Read, Show)

instance Monoid (Either [MakefileError] Makefile) where
  mempty = Right (Makefile mempty mempty)
  mappend mx my = do
    Makefile x1 x2 <- mx
    Makefile y1 y2 <- my
    z1 <- left (ERuleConflict <$>) (unionEqMap x1 y1)
    z2 <- left (EMacroConflict <$>) (unionEqMap x2 y2)
    pure (Makefile z1 z2)

type Make a = Writer (Makefile, VarMap) a

-- | Laws are @isMempty mempty ≡ True@ and if @a ≡ t b@ and @Foldable t@ then
-- @isMempty ≡ null@.
class Monoid a => MemptyComparable a where
  isMempty :: a -> Bool

data Mk m a = Mk m a
