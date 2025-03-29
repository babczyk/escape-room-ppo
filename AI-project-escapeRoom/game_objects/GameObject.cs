using System;
using Microsoft.Xna.Framework;
using Microsoft.Xna.Framework.Graphics;
public abstract class GameObject
{
    public Vector2 Position { get; set; }
    public Vector2 Velocity { get; set; }
    public Vector2 Size { get; set; }
    public bool IsGrounded { get; set; }
    public String ROLL { get; set; }

    public float gravity { private get; set; }
    private float terminalVelocity;
    private Texture2D texture;

    public GameObject(Vector2 position, Vector2 size, String roll = "ROLL", float gravity = 9.8f, float terminalVelocity = 100000f)
    {
        Position = position;
        Velocity = Vector2.Zero;
        Size = size;
        this.gravity = gravity;
        this.terminalVelocity = terminalVelocity;
        this.ROLL = roll;
    }

    public void LoadTexture(GraphicsDevice graphicsDevice, Color color)
    {
        // Create a simple colored texture for the object
        texture = new Texture2D(graphicsDevice, (int)Size.X, (int)Size.Y);
        Color[] data = new Color[(int)Size.X * (int)Size.Y];
        for (int i = 0; i < data.Length; ++i) data[i] = color;
        texture.SetData(data);
    }

    public void Update(GameTime gameTime)
    {
        float deltaTime = (float)gameTime.ElapsedGameTime.TotalSeconds;

        // Apply gravity if not grounded
        if (!IsGrounded)
        {
            Velocity += new Vector2(0, gravity);

            // Clamp velocity to terminal velocity
            if (Velocity.Y > terminalVelocity)
                Velocity = new Vector2(Velocity.X, terminalVelocity);
        }

        // Update position
        Position += Velocity * deltaTime;

    }

    public void Draw(SpriteBatch spriteBatch)
    {
        if (texture != null)
        {
            spriteBatch.Draw(texture, Position, Color.White);
        }
    }

    public void ApplyForce(Vector2 force)
    {
        Velocity += force;
    }
    public void ApplyImpulse(Vector2 impulse)
    {
        Velocity = impulse;
    }
    public void Move(Vector2 move)
    {
        Position += move;
    }
    public bool Intersects(GameObject other)
    {
        int bottomOverlap = 0; // Extra overlap on the bottom to prevent sticking to the ground
        // Create a rectangle for this object
        Rectangle thisRect = new Rectangle((int)Position.X, (int)Position.Y, (int)Size.X, (int)Size.Y + (int)bottomOverlap);

        // Create a rectangle for the other object
        Rectangle otherRect = new Rectangle((int)other.Position.X, (int)other.Position.Y, (int)other.Size.X, (int)other.Size.Y);

        // Check for intersection manually
        bool intersects = thisRect.Left <= otherRect.Right &&
              thisRect.Right >= otherRect.Left &&
              thisRect.Top <= otherRect.Bottom &&
              thisRect.Bottom >= otherRect.Top;
        //System.Console.WriteLine(thisRect.Bottom + " " + otherRect.Top + " " + intersects + " " + ROLL);

        // Check for intersection
        return intersects;
    }


    public void StopByWalls(Wall[] walls)
    {
        Wall closestWall = null;
        float smallestOverlap = float.MaxValue;

        foreach (var wall in walls)
        {
            if (Intersects(wall))
            {
                // Calculate the overlap distances
                float overlapX = Math.Min(Position.X + Size.X - wall.Position.X, wall.Position.X + wall.Size.X - Position.X);
                float overlapY = Math.Min(Position.Y + Size.Y - wall.Position.Y, wall.Position.Y + wall.Size.Y - Position.Y);

                // Determine the smallest overlap
                float overlap = Math.Min(overlapX, overlapY);

                if (overlap < smallestOverlap)
                {
                    smallestOverlap = overlap;
                    closestWall = wall;
                }
            }
        }

        // Resolve collision with the closest wall, if any
        if (closestWall != null)
        {
            float overlapX = Math.Min(Position.X + Size.X - closestWall.Position.X, closestWall.Position.X + closestWall.Size.X - Position.X);
            float overlapY = Math.Min(Position.Y + Size.Y - closestWall.Position.Y, closestWall.Position.Y + closestWall.Size.Y - Position.Y);

            if (overlapX < overlapY)
            {
                // Resolve X-axis collision
                if (closestWall.Position.X < Position.X)
                    Position = new Vector2(closestWall.Position.X + closestWall.Size.X, Position.Y);
                else
                    Position = new Vector2(closestWall.Position.X - Size.X, Position.Y);

                // Stop horizontal movement
                Velocity = new Vector2(0, Velocity.Y);
            }
            else
            {
                // Resolve Y-axis collision
                if (closestWall.Position.Y < Position.Y)
                {
                    Position = new Vector2(Position.X, closestWall.Position.Y + closestWall.Size.Y);
                }
                else
                {
                    if (Velocity.Y > 0)
                        Velocity = new Vector2(Velocity.X, 0);

                    Position = new Vector2(Position.X, closestWall.Position.Y - Size.Y);
                    IsGrounded = true;
                }
            }
        }
        else
        {
            IsGrounded = false; // No collision detected
        }
    }

}
